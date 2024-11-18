import time
import random
import logging
import collections
import numpy as np
import lightgbm as lgb
from randomdict import RandomDict
from configparser import ConfigParser
from caches.cache_base import CacheBase, Request


class Meta(object):
    def __init__(self, seq, id, size):
        self.past_timestamp = seq
        self.id  = id
        self.size = size
        self.sample_times = []
        self.past_distances = []
        self.max_distance = -1
        self.min_distance = -1
        self.mean_distance = 9999999
        self.freq = 0

    def update(self, timestamp):
        past_distance = timestamp - self.past_timestamp
        self.past_distances.append(past_distance)
        self.past_timestamp = timestamp
        if len(self.past_distances) == 1:
            self.max_distance = past_distance
            self.min_distance = past_distance
            self.mean_distance = past_distance
        else:
            self.max_distance = max(self.max_distance, past_distance)
            self.min_distance = min(self.min_distance, past_distance)
            self.mean_distance = (self.mean_distance * self.freq + past_distance) / (self.freq + 1)
        self.freq += 1

    def sample(self, timestamp):
        self.sample_times.append(timestamp)

    def clear_sample(self):
        self.sample_times = []


class TrainingData(object):
    def __init__(self, memory_window, max_n_past_timestamps, use_size, use_n_within, use_edc):
        self.memory_window = memory_window
        self.max_n_past_timestamps = max_n_past_timestamps
        self.max_n_past_distances = max_n_past_timestamps - 1
        self.use_size = use_size
        self.use_n_within = use_n_within
        self.use_edc = use_edc

        self.data = []
        self.labels = []

    def add(self, meta: Meta, sample_timestamp: int, future_interval: int):
        feats = []
        feats.append(sample_timestamp - meta.past_timestamp)
        dists = meta.past_distances[-self.max_n_past_distances:]
        feats.extend(dists)
        if len(dists) < self.max_n_past_distances:
            feats.extend([0] * (self.max_n_past_distances - len(dists)))
        if self.use_size:
            feats.append(meta.size)
        if self.use_n_within:
            this_past_distance = 0
            n_within = 0
            for dist in reversed(dists):
                this_past_distance += dist
                if this_past_distance < self.memory_window:
                    n_within += 1
            feats.append(n_within)
        if self.use_edc:
            # FIXME: add use_edc
            raise Exception("edc features not supported yet")
        self.data.append(feats)
        self.labels.append(np.log1p(future_interval))

    def clear(self):
        self.data = []
        self.labels = []


class LRBCache(CacheBase):
    def __init__(self, capacity, config: ConfigParser):
        super().__init__(capacity)
        # parameters
        self.sample_size = config.getint('lrb', 'sample_size', fallback=64)
        self.batch_size = config.getint('lrb', 'batch_size', fallback=131072)
        self.memory_window = config.getint('lrb', 'memory_window', fallback=1000000)
        self.max_n_past_timestamps = config.getint('lrb', 'max_n_past_timestamps', fallback=32)
        self.max_n_past_distances = self.max_n_past_timestamps - 1
        self.num_iterations = config.getint('lrb', 'num_iterations', fallback=32)
        self.priority_size_function = config.get('lrb', 'priority_size_function', fallback='identity')
        self.use_size = config.getboolean('lrb', 'use_size', fallback=False)
        self.use_edc = config.getboolean('lrb', 'use_edc', fallback=False)
        self.use_n_within = config.getboolean('lrb', 'use_n_within', fallback=False)
        self.use_seq_for_retrain = config.getboolean('lrb', 'use_seq_for_retrain', fallback=False)
        self.use_noforget = config.getboolean('lrb', 'use_noforget', fallback=False)

        # data structures
        self.current_seq = -1
        self.is_sampling = False
        self.ml_model = None
        # maps object id to request and maintain order
        self.lru_cache = collections.OrderedDict()
        # maps timestamp to request
        self.forget_candidate = {}
        # maps object id to in cache meta
        self.in_cache_metas = RandomDict()
        # maps object id to out cache meta
        self.out_cache_metas = RandomDict()
        self.training_data = TrainingData(
            memory_window=self.memory_window, max_n_past_timestamps=self.max_n_past_timestamps,
            use_size=self.use_size, use_n_within=self.use_n_within, use_edc=self.use_edc)

        # statistics
        self.n_force_eviction = 0
        self.window_index = -1
        self.warm_index = -1

        # runtime counter
        self.ml_training_time = 0
        self.admitting_time = 0
        self.admitting_count = 0
        self.evicting_time = 0
        self.evicting_count = 0


    def lookup(self, req: Request) -> bool:
        self.current_seq += 1
        hit = super().lookup(req=req)
        # maintain the order
        if hit:
            self.lru_cache.move_to_end(req.id)
        self.forget()
        # manipulate meta
        if req.id in self.in_cache_metas or req.id in self.out_cache_metas:
            if req.id in self.in_cache_metas:
                assert hit
                meta = self.in_cache_metas[req.id]
            else:
                assert not hit
                meta = self.out_cache_metas[req.id]
            last_timestamp = meta.past_timestamp
            forget_timestamp = last_timestamp % self.memory_window
            # if the key in out_metadata, it must also in forget table
            assert hit or forget_timestamp in self.forget_candidate
            # re-request
            if len(meta.sample_times) > 0:
                for sample_time in meta.sample_times:
                    future_distance = self.current_seq - sample_time
                    self.training_data.add(meta=meta, sample_timestamp=sample_time, future_interval=future_distance)
                if not self.use_seq_for_retrain and len(self.training_data.labels) >= self.batch_size:
                    self.train()
                    self.training_data.clear()
                meta.clear_sample()
            meta.update(timestamp=self.current_seq)
            if req.id in self.out_cache_metas:
                self.forget_candidate.pop(forget_timestamp)
                self.forget_candidate[self.current_seq % self.memory_window] = req
        else:
            assert not hit
        # sampling happens late to prevent immediate re-request
        if self.is_sampling:
            self.sampling()
        # retrain if using sequence
        if self.use_seq_for_retrain and self.current_seq > 0 and self.current_seq % self.batch_size == 0:
            ml_train_start_time = time.time()
            self.train()
            self.training_data.clear()
            self.ml_training_time = time.time() - ml_train_start_time
        return hit

    def admit(self, req: Request):
        if self.ml_model != None:
            admit_start_time = time.time()
        if req.size > self.capacity:
            logging.error("Object size %s is larger than cache size %s, cannot admit", req.size, self.capacity)
            return
        super().admit(req)
        # maintain the order
        self.lru_cache[req.id] = req
        self.lru_cache.move_to_end(req.id)
        # manipulate meta
        if req.id in self.in_cache_metas or req.id in self.out_cache_metas:
            # must be in out cache meta, bring from out to in
            assert req.id not in self.in_cache_metas
            meta = self.out_cache_metas.pop(req.id)
            forget_timestamp = meta.past_timestamp % self.memory_window
            self.forget_candidate.pop(forget_timestamp)
            self.in_cache_metas[req.id] = meta
        else:
            # fresh insert
            self.in_cache_metas[req.id] = Meta(seq=self.current_seq, id=req.id, size=req.size)
        if self.remain < 0:
            self.is_sampling = True
        if self.ml_model != None:
            admit_end_time = time.time()
            self.admitting_count += 1
            self.admitting_time = admit_end_time - admit_start_time
        while self.remain < 0:
            self.evict(None)

    def evict(self, obj):
        if self.ml_model != None:
            evict_start_time = time.time()
        dobj, dreq = self.rank()
        meta = self.in_cache_metas.pop(dobj)
        if self.current_seq - meta.past_timestamp >= self.memory_window:
            # must be the tail of lru
            if len(meta.sample_times) > 0:
                future_distance = self.current_seq - meta.past_timestamp + self.memory_window
                for sample_time in meta.sample_times:
                    self.training_data.add(meta=meta, sample_timestamp=sample_time, future_interval=future_distance)
                if not self.use_seq_for_retrain and len(self.training_data.labels) >= self.batch_size:
                    self.train()
                    self.training_data.clear()
            self.n_force_eviction += 1
        else:
            # must be in in cache meta, bring from in to out
            self.out_cache_metas[dobj] = meta
            self.forget_candidate[meta.past_timestamp % self.memory_window] = dreq

        super().evict(dobj)
        self.lru_cache.pop(dobj)
        if self.ml_model != None:
            evict_end_time = time.time()
            self.evicting_count += 1
            self.evicting_time = evict_end_time - evict_start_time

    def forget(self):
        if self.use_noforget:
            return
        forget_timestamp = self.current_seq % self.memory_window
        if forget_timestamp in self.forget_candidate:
            forget_obj = self.forget_candidate[forget_timestamp].id
            meta = self.out_cache_metas.pop(forget_obj)
            if len(meta.sample_times) > 0:
                future_distance = self.memory_window * 2
                for sample_time in meta.sample_times:
                    self.training_data.add(meta=meta, sample_timestamp=sample_time, future_interval=future_distance)
                if not self.use_seq_for_retrain and len(self.training_data.labels) >= self.batch_size:
                    self.train()
                    self.training_data.clear()
            self.forget_candidate.pop(forget_timestamp)

    def sampling(self):
        index = random.randint(0, len(self.in_cache_metas) + len(self.out_cache_metas)-1)
        if index < len(self.in_cache_metas):
            meta = self.in_cache_metas.random_value()
        else:
            meta = self.out_cache_metas.random_value()
        meta.sample(timestamp=self.current_seq)

    def _prepare_data(self, meta: Meta):
        feats = []
        feats.append(self.current_seq - meta.past_timestamp)
        dists = meta.past_distances[-self.max_n_past_distances:]
        feats.extend(dists)
        if len(dists) < self.max_n_past_distances:
            feats.extend([0] * (self.max_n_past_distances - len(dists)))
        if self.use_size:
            feats.append(meta.size)
        if self.use_n_within:
            this_past_distance = 0
            n_within = 0
            for dist in reversed(dists):
                this_past_distance += dist
                if this_past_distance < self.memory_window:
                    n_within += 1
            feats.append(n_within)
        if self.use_edc:
            # FIXME: add use_edc
            raise Exception("edc features not supported yet")
        return feats

    def rank(self):
        # if not trained yet, or in_cache_lru past memory window, use LRU
        candidate = next(iter(self.lru_cache))
        meta = self.in_cache_metas[candidate]
        if not self.ml_model or self.current_seq - meta.past_timestamp >= self.memory_window:
            return meta.id, self.cache_memory[meta.id]

        # sample objects and compute their future intervals
        pred_objs = self.sample(self.sample_size)
        pred_data = []
        for pred_obj in pred_objs:
            pred_meta = self.in_cache_metas[pred_obj]
            pred_data.append(self._prepare_data(meta=pred_meta))
        pred_data = np.array(pred_data)
        pred_distances = self.ml_model.predict(pred_data, num_iteration=self.ml_model.best_iteration)
        if self.priority_size_function != 'identity':
            raise Exception("priority size function: %s not implemented yet!" % self.priority_size_function)
        max_obj_index = pred_distances.argmax()
        dobj = pred_objs[max_obj_index]
        return dobj, self.cache_memory[dobj]

    def train(self):
        if self.window_index == -1:
            self.warm_index = self.current_seq
        self.window_index += 1

        training_start = time.time()
        logging.info("training booster at %s", self.current_seq)
        lgb_train = lgb.Dataset(np.array(self.training_data.data), self.training_data.labels)

        # specify your configurations as a dict
        params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': {'l2'},
            'num_iterations': self.num_iterations,
            'num_leaves': 32,
            'num_threads': 4,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 0
        }
        # train
        self.ml_model = lgb.train(
            params, lgb_train, num_boost_round=32, valid_sets=lgb_train,
            callbacks=[lgb.early_stopping(stopping_rounds=5)])
        logging.info("finished training booster at %s took %s seconds", self.current_seq, time.time() - training_start)

import time
import logging
import numpy as np
from collections import Counter
from caches.cache_base import Request
from caches.lrb import LRBCache


class HeuristicModel(object):
    def __init__(self, window_mean_distance, memory_window, dist_sample_count, rank_metric, max_lookback, consider_decay):
        self.window_mean_distance = window_mean_distance
        self.memory_window = memory_window
        self.dist_sample_count = dist_sample_count
        self.rank_metric = rank_metric
        self.max_lookback = max_lookback
        self.consider_decay = consider_decay

    def predict(self, current_seq, metas):
        # get valid samples from each meta
        residual_times = []
        for meta in metas:
            dobj_age = current_seq - meta.past_timestamp
            if len(meta.past_distances) > 0:
                if dobj_age < meta.min_distance:
                    valid_sample = np.array(meta.past_distances[-self.dist_sample_count:]) - dobj_age
                    if len(valid_sample) < self.dist_sample_count:
                        valid_sample = np.resize(valid_sample, (self.dist_sample_count,))
                    residual_times.append(valid_sample)
                elif dobj_age >= meta.max_distance:
                    if not self.consider_decay:
                        if dobj_age >= self.memory_window:
                            residual_times.append(np.array([dobj_age + self.memory_window] * self.dist_sample_count))
                        else:
                            residual_times.append(np.array([dobj_age + self.window_mean_distance] * self.dist_sample_count))
                    else:
                        if dobj_age >= self.memory_window:
                            residual_times.append(np.array([(dobj_age + self.memory_window) / len(meta.past_distances)] * self.dist_sample_count))
                        else:
                            residual_times.append(np.array([(dobj_age + self.window_mean_distance) / len(meta.past_distances)] * self.dist_sample_count))
                else:
                    samples = np.array(meta.past_distances[-self.max_lookback:])
                    valid_sample_idx = samples > dobj_age
                    valid_sample = samples[valid_sample_idx][-self.dist_sample_count:] - dobj_age
                    if len(valid_sample) < self.dist_sample_count:
                        valid_sample = np.resize(valid_sample, (self.dist_sample_count,))
                    residual_times.append(valid_sample)
            else:
                # handle new objects that are requested
                if dobj_age >= self.memory_window:
                    residual_times.append(np.array([dobj_age + self.memory_window] * self.dist_sample_count))
                else:
                    residual_times.append(np.array([dobj_age + self.window_mean_distance] * self.dist_sample_count))
        residual_times = np.array(residual_times)
        if self.rank_metric == 'sample':
            max_residual_idx = residual_times.argmax(axis=0)
            # meta_idx = np.argmax(np.bincount(max_residual_idx, minlength=len(metas)))
            meta_idx = Counter(max_residual_idx).most_common(1)[0][0]
        elif self.rank_metric == 'mean':
            meta_idx = np.mean(residual_times, axis=1).argmax()
        elif self.rank_metric == 'median':
            meta_idx = np.median(residual_times, axis=1).argmax()
        else:
            raise Exception("Unexpected rank metric: %s" % self.rank_metric)
        return metas[meta_idx].id 


class RavenHeuristic(LRBCache):
    def __init__(self, capacity, config):
        super().__init__(capacity=capacity, config=config)
        # parameters
        self.sample_size = config.getint('ravenh', 'sample_size', fallback=64)
        self.batch_size = config.getint('ravenh', 'batch_size', fallback=131072)
        self.memory_window = config.getint('ravenh', 'memory_window', fallback=1000000)
        self.max_n_past_timestamps = config.getint('ravenh', 'max_n_past_timestamps', fallback=32)
        self.max_n_past_distances = self.max_n_past_timestamps - 1
        self.num_iterations = config.getint('ravenh', 'num_iterations', fallback=32)
        self.priority_size_function = config.get('ravenh', 'priority_size_function', fallback='identity')
        # skip training parameters in LRB, because they are not used
        self.use_seq_for_retrain = config.getboolean('ravenh', 'use_seq_for_retrain', fallback=False)
        self.use_noforget = config.getboolean('ravenh', 'use_noforget', fallback=False)

        # raven heuristic specific parameters
        self.warmup_policy = config.get('ravenh', 'warmup_policy', fallback='lru')
        self.dist_sample_count = config.getint('ravenh', 'dist_sample_count', fallback=100)
        self.rank_metric = config.get('ravenh', 'rank_metric', fallback='sample')
        self.max_lookback = config.getint('ravenh', 'max_lookback', fallback=10000)
        self.consider_decay = config.getboolean('ravenh', 'consider_decay', fallback=False)
        self.window_mean_distance = self.memory_window / 2
        self.window_index = -1
        self.warm_index = -1

    def rank(self):
        # if not trained yet and warm up policy is lru, use lru
        candidate = next(iter(self.lru_cache))
        meta = self.in_cache_metas[candidate]
        if (self.warmup_policy == 'lru' and not self.ml_model) or self.current_seq - meta.past_timestamp >= self.memory_window:
            return meta.id, self.cache_memory[meta.id]
        # if not trained yet and warm up policy is ravenh, use ravenh
        if (self.warmup_policy == 'ravenh' and not self.ml_model):
            self.ml_model = HeuristicModel(
                window_mean_distance=self.window_mean_distance, memory_window=self.memory_window,
                dist_sample_count=self.dist_sample_count, rank_metric=self.rank_metric, max_lookback=self.max_lookback,
                consider_decay=self.consider_decay)
        pred_metas = [self.in_cache_metas[pred_obj] for pred_obj in self.sample(self.sample_size)]
        dobj = self.ml_model.predict(current_seq=self.current_seq, metas=pred_metas)
        return dobj, self.cache_memory[dobj]

    def train(self):
        if self.window_index == -1:
            self.warm_index = self.current_seq
        self.window_index += 1

        training_start = time.time()
        logging.info("computing window mean tao at %s", self.current_seq)
        window_distance_sum = 0
        window_obj_count = 0
        for _, meta in self.in_cache_metas.items():
            if len(meta.past_distances) == 0:
                continue
            meta.past_distances = meta.past_distances[-self.max_lookback:]
            window_distance_sum += np.mean(meta.past_distances)
            window_obj_count += 1
        for _, meta in self.out_cache_metas.items():
            if len(meta.past_distances) == 0:
                continue
            meta.past_distances = meta.past_distances[-self.max_lookback:]
            window_distance_sum += np.mean(meta.past_distances)
            window_obj_count += 1
        self.window_mean_distance = window_distance_sum / window_obj_count
        self.ml_model = HeuristicModel(
            window_mean_distance=self.window_mean_distance, memory_window=self.memory_window,
            dist_sample_count=self.dist_sample_count, rank_metric=self.rank_metric, max_lookback=self.max_lookback,
            consider_decay=self.consider_decay)
        logging.info("computed the window mean distance at %s to be %s for %s seconds",
                     self.current_seq, self.window_mean_distance, time.time() - training_start)

import os
import logging
import time
import torch
import random
import collections
import numpy as np
from randomdict import RandomDict
from copy import deepcopy
from collections import Counter
from os.path import join, splitext, basename, exists
from caches.lrb import LRBCache, Meta
from configparser import ConfigParser
from models.sequence import Sequence
from models.dataset import SequenceDataset
from models.batch import Batch
from models.tpp_residual import RecurrentTPPResidual
from models.tpp_inter import RecurrentTPPInter
from caches.cache_base import CacheBase, Request


if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


class RavenLearnTao(CacheBase):
    def __init__(self, capacity, config: ConfigParser):
        super().__init__(capacity=capacity)
        # parameters
        # overwrite LRB
        self.sample_size = config.getint('ravenl', 'sample_size', fallback=64)
        # todo: think about when to train ML
        self.batch_size = config.getint('ravenl', 'batch_size', fallback=131072)
        self.memory_window = config.getint('ravenl', 'memory_window', fallback=1000000)
        # A reasonable limit of 250-500 time steps is often used in practice with large LSTM models.
        self.max_n_past_timestamps = config.getint('ravenl', 'max_n_past_timestamps', fallback=32)
        self.max_n_past_distances = self.max_n_past_timestamps - 1
        self.priority_size_function = config.get('ravenl', 'priority_size_function', fallback='identity')
        self.use_size = config.getboolean('ravenl', 'use_size', fallback=False)
        self.use_edc = config.getboolean('ravenl', 'use_edc', fallback=False)
        self.use_n_within = config.getboolean('ravenl', 'use_n_within', fallback=False)
        # Raven Learn unique
        self.learn_objective = config.get('ravenl', 'learn_objective', fallback='inter')
        # train_data_type: latest, all
        self.train_data_type = config.get('ravenl', 'train_data_type', fallback='latest')
        # minimum seq len to put into training data
        self.min_n_past_distances = config.getint('ravenl', 'min_n_past_distances', fallback=10)
        self.dist_sample_size = config.getint('ravenl', 'dist_sample_size', fallback=10)
        self.history_type = config.get('ravenl', 'history_type', fallback='lrb')
        self.load_ml = config.getboolean('ravenl', 'load_ml', fallback=False)
        self.context_size = config.getint('ravenl', 'context_size', fallback=32)
        self.rnn_type = config.get('ravenl', 'rnn_type', fallback='GRU')
        self.ml_batch_size = config.getint('ravenl', 'ml_batch_size', fallback=256)
        self.reload = config.getboolean('ravenl', 'reload', fallback=False)
        self.savedir = config.get('ravenl', 'savedir', fallback='./ckpoints_{trace}_{tag}').format(trace=config.trace, tag=config.tag)
        self.savedir = self.savedir.split('#')[0]
        self.savename = join(self.savedir, config.get('ravenl', 'savename', fallback='model_{window_index}'))
        if not exists(self.savedir):
            os.makedirs(self.savedir)
        self.consider_survival = config.getboolean('ravenl', 'consider_survival', fallback=False)
        self.consider_objid = config.getboolean('ravenl', 'consider_objid', fallback=True)
        self.consider_objsize = config.getboolean('ravenl', 'consider_objsize', fallback=True)
        self.use_pseudo_sample = config.getboolean('ravenl', 'use_pseudo_sample', fallback=True)
        self.rank_criteria = config.get('ravenl', 'rank_criteria', fallback='sample')

        self.cuda = config.getint('ravenl', 'cuda', fallback=0)
        if torch.cuda.is_available():
            torch.cuda.set_device(self.cuda)

        # ML learning
        self.num_features = 1  # Input feature to rnn
        self.n_components = 64
        self.regularization = 1e-5
        self.lr = 1e-3
        self.num_iterations = 1000
        self.display_step = 20
        self.patience = 200
        self.seed = 0

        # data structures
        self.window_index = -1
        self.warm_index = -1
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
        self.n_force_eviction = 0
        self.n_random_guess = 0
        self.max_group_id = 0
        self.mean_obj_size = 0
        self.std_obj_size = 1.0

        # runtime counter
        self.ml_training_time = 0
        self.admitting_time = 0
        self.admitting_count = 0
        self.evicting_time = 0
        self.evicting_count = 0

        # load_ml and re_load cannot be True at the same time
        assert not (self.load_ml == True and self.reload == True)
        if self.priority_size_function !='identity' and self.priority_size_function !='linear':
            raise Exception("priority size function: %s not implemented yet!" % self.priority_size_function)

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
            meta.update(timestamp=self.current_seq)
            if req.id in self.out_cache_metas:
                self.forget_candidate.pop(forget_timestamp)
                self.forget_candidate[self.current_seq % self.memory_window] = req
        else:
            assert not hit
        if self.current_seq % self.batch_size == 0 and self.current_seq > 0:
            ml_train_start_time = time.time()
            self.train()
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
            # todo: add additional training sample
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
        forget_timestamp = self.current_seq % self.memory_window
        if forget_timestamp in self.forget_candidate:
            # todo: add additional training sample
            forget_obj = self.forget_candidate[forget_timestamp].id
            meta = self.out_cache_metas.pop(forget_obj)
            self.forget_candidate.pop(forget_timestamp)

    def rank(self):
        # if not trained yet, or in_cache_lru past memory window, use LRU
        candidate = next(iter(self.lru_cache))
        meta = self.in_cache_metas[candidate]
        if not self.ml_model or self.current_seq - meta.past_timestamp >= self.memory_window:
            return meta.id, self.cache_memory[meta.id]

        # sample objects and compute their future intervals
        pred_objs = self.sample(self.sample_size)
        pred_data = []
        pred_objids = []
        pred_age = []
        pred_last_event_idx = []
        pred_objsizes = []

        for pred_obj in pred_objs:
            pred_meta = self.in_cache_metas[pred_obj]
            dists = pred_meta.past_distances[-self.max_n_past_distances:]
            pred_last_event_idx.append(len(dists) - 1)
            dists =  dists + [0] * (self.max_n_past_distances - len(dists))
            pred_data.append(dists)
            pred_objids.append(pred_meta.mean_distance)
            #pred_objsizes.append((pred_meta.size - self.mean_obj_size) / self.std_obj_size)
            pred_objsizes.append(pred_meta.size)
            pred_age.append(self.current_seq - pred_meta.past_timestamp)
        pred_objsizes = np.array(pred_objsizes)
        pred_data = np.array(pred_data)
        pred_data_batch = Batch(inter_times=torch.Tensor(pred_data), mask=torch.ones_like(torch.Tensor(pred_data)))

        with torch.no_grad():
            features = self.ml_model.get_features(pred_data_batch)
            context = self.ml_model.get_context(features, remove_last=False)  # (batch_size, seq_len, context_size)
            context = context[[idx for idx in range(len(pred_last_event_idx))], pred_last_event_idx, :]  # (batch_size, context_size)

            if self.learn_objective == 'residual':
                pred_age = torch.Tensor(np.array(pred_age)).unsqueeze(-1)
                obj_ages = torch.log(pred_age + 1e-8)
                obj_ages = (obj_ages - self.ml_model.mean_log_inter_time) / self.ml_model.std_log_inter_time
                context = torch.cat([context, obj_ages], dim=-1)
            if self.consider_objid:
                pred_objids = torch.Tensor(pred_objids).unsqueeze(-1)  # (batch_size, 1)
                pred_objids = torch.log(pred_objids + 1e-8)
                pred_objids = (pred_objids - self.ml_model.mean_log_inter_time) / self.ml_model.std_log_inter_time
                context = torch.cat([context, pred_objids], dim=-1)
            if self.consider_objsize:
                normalized_pred_objsizes = torch.Tensor((pred_objsizes - self.mean_obj_size) / self.std_obj_size).unsqueeze(-1)  # (batch_size, 1)
                context = torch.cat([context, normalized_pred_objsizes], dim=-1)

            if self.learn_objective == 'residual':
                # obj_ages = torch.log(pred_age + 1e-8)
                # obj_ages = (obj_ages - self.ml_model.mean_log_inter_time) / self.ml_model.std_log_inter_time
                # obj_ages = obj_ages.unsqueeze(-1)
                # new_context = torch.cat([context, obj_ages], dim=-1)
                residual_time_dist = self.ml_model.get_residual_time_dist(context)
                if self.rank_criteria == 'sample':
                    residual_time_samples = residual_time_dist.sample((self.dist_sample_size,)).cpu().numpy()
                elif self.rank_criteria == 'median':
                    residual_time_samples = residual_time_dist.sample((self.dist_sample_size,)).cpu().numpy()
                    residual_time_samples = np.median(residual_time_samples, axis=0)
                elif self.rank_criteria == 'mean':
                    residual_time_samples = residual_time_dist.mean.detach().cpu().numpy()
                else:
                    raise Exception('undefined rank criteria %s' % self.rank_criteria)
                if self.priority_size_function == 'linear':
                    residual_time_samples = residual_time_samples * pred_objsizes
            elif self.learn_objective == 'inter':
                inter_time_dist = self.ml_model.get_inter_time_dist(context)
                inter_time_samples = inter_time_dist.sample((1000,)).cpu().numpy()  # sample_size * object number
                residual_time_samples = []
                for obj_idx in range(len(pred_objs)):
                    valid_sample = inter_time_samples[:, obj_idx][inter_time_samples[:, obj_idx] > pred_age[obj_idx]]
                    if len(valid_sample) <= 0:
                        valid_sample = np.array([(pred_age[obj_idx] + 2 * self.memory_window)] * self.dist_sample_size)
                        if self.priority_size_function == 'linear':
                            valid_sample = valid_sample * pred_objsizes[obj_idx]
                        residual_time_samples.append(valid_sample)
                        self.n_random_guess += 1
                    else:
                        if len(valid_sample) < self.dist_sample_size:
                            valid_sample = np.resize(valid_sample, (self.dist_sample_size,))
                        valid_sample = valid_sample[-self.dist_sample_size:] - pred_age[obj_idx]
                        if self.priority_size_function == 'linear':
                            valid_sample = valid_sample * pred_objsizes[obj_idx]
                        residual_time_samples.append(valid_sample)
                residual_time_samples = np.array(residual_time_samples)
            else:
                raise Exception('undefined learning objective %s' % self.learn_objective)

        if self.learn_objective == 'residual':
            if self.rank_criteria == 'sample':
                max_residual_idx = np.argmax(residual_time_samples, axis=1)
                meta_idx = Counter(max_residual_idx).most_common(1)[0][0]
            else:
                meta_idx = np.argmax(residual_time_samples)
        elif self.learn_objective == 'inter':
            max_residual_idx = np.argmax(residual_time_samples, axis=0)
            meta_idx = Counter(max_residual_idx).most_common(1)[0][0]
        dobj = pred_objs[meta_idx]
        return dobj, self.cache_memory[dobj]

    def _density_nn_dataset(self):
        sequences_train = []
        training_obj_sizes = []
        for _, meta in self.in_cache_metas.items():
            training_obj_sizes.append(meta.size)
        for _, meta in self.out_cache_metas.items():
            training_obj_sizes.append(meta.size)
        self.mean_obj_size = np.mean(training_obj_sizes)
        self.std_obj_size = np.std(training_obj_sizes)

        for _, meta in self.in_cache_metas.items():
            pseudo_sample = -1
            if self.use_pseudo_sample:
                age = self.current_seq - meta.past_timestamp
                #meta_std = np.std(meta.past_distances)
                #meta_mean = np.mean(meta.past_distances)
                if age >= 1000000:
                    #pseudo_sample = age + meta_mean + 3 * meta_std
                    pseudo_sample = age + 1000000

            if self.train_data_type == 'latest':
                if len(meta.past_distances) == 0:
                    continue
                if self.use_pseudo_sample and pseudo_sample > 0 and not self.consider_survival:
                    seq = meta.past_distances[-self.max_n_past_distances: ] + [pseudo_sample, self.current_seq - meta.past_timestamp]
                else:
                    seq = meta.past_distances[-self.max_n_past_distances: ] + [self.current_seq - meta.past_timestamp]
                # in dpp data structure, last is always survival and is masked.
                normalized_meta_size = (meta.size - self.mean_obj_size) / self.std_obj_size
                if self.consider_objid and not self.consider_objsize:
                    sequences_train.append(Sequence(inter_times=seq, obj_id=meta.mean_distance))
                if not self.consider_objid and self.consider_objsize:
                    sequences_train.append(Sequence(inter_times=seq, obj_size=normalized_meta_size))
                if self.consider_objid and self.consider_objsize:
                    sequences_train.append(Sequence(inter_times=seq, obj_id=meta.mean_distance, obj_size=normalized_meta_size))
                if not self.consider_objid and not self.consider_objsize:
                    sequences_train.append(Sequence(inter_times=seq))

            elif self.train_data_type == 'all':
                num_batches = int(np.ceil(len(meta.past_distances) / self.max_n_past_distances))
                for b in range(num_batches):
                    if b == num_batches - 1:
                        if self.use_pseudo_sample and pseudo_sample > 0 and not self.consider_survival:
                            seq = meta.past_distances[
                                  b * self.max_n_past_distances: (b + 1) * self.max_n_past_distances] + [
                                      pseudo_sample, self.current_seq - meta.past_timestamp]
                        else:
                            seq = meta.past_distances[b * self.max_n_past_distances: (b + 1) * self.max_n_past_distances] + [self.current_seq - meta.past_timestamp]
                    else:
                        seq = meta.past_distances[b * self.max_n_past_distances: (b + 1) * self.max_n_past_distances] + \
                                [np.random.uniform(meta.past_distances[(b + 1) * self.max_n_past_distances])]
                    if len(seq) == 1:
                        continue
                    normalized_meta_size = (meta.size - self.mean_obj_size) / self.std_obj_size
                    if self.consider_objid and not self.consider_objsize:
                        sequences_train.append(Sequence(inter_times=seq, obj_id=meta.mean_distance))
                    if not self.consider_objid and self.consider_objsize:
                        sequences_train.append(Sequence(inter_times=seq, obj_size=normalized_meta_size))
                    if self.consider_objid and self.consider_objsize:
                        sequences_train.append(Sequence(inter_times=seq, obj_id=meta.mean_distance, obj_size=normalized_meta_size))
                    if not self.consider_objid and not self.consider_objsize:
                        sequences_train.append(Sequence(inter_times=seq))
        for _, meta in self.out_cache_metas.items():
            pseudo_sample = -1
            if self.use_pseudo_sample:
                age = self.current_seq - meta.past_timestamp
                # meta_std = np.std(meta.past_distances)
                # meta_mean = np.mean(meta.past_distances)
                if age >= 1000000:
                    # pseudo_sample = age + meta_mean + 3 * meta_std
                    pseudo_sample = age + 1000000

            if self.train_data_type == 'latest':
                if len(meta.past_distances) == 0:
                    continue
                if self.use_pseudo_sample and pseudo_sample > 0 and not self.consider_survival:
                    seq = meta.past_distances[-self.max_n_past_distances: ] + [pseudo_sample, self.current_seq - meta.past_timestamp]
                else:
                    seq = meta.past_distances[-self.max_n_past_distances: ] + [self.current_seq - meta.past_timestamp]
                # in dpp data structure, last is always survival and is masked.
                normalized_meta_size = (meta.size - self.mean_obj_size) / self.std_obj_size
                if self.consider_objid and not self.consider_objsize:
                    sequences_train.append(Sequence(inter_times=seq, obj_id=meta.mean_distance))
                if not self.consider_objid and self.consider_objsize:
                    sequences_train.append(Sequence(inter_times=seq, obj_size=normalized_meta_size))
                if self.consider_objid and self.consider_objsize:
                    sequences_train.append(Sequence(inter_times=seq, obj_id=meta.mean_distance, obj_size=normalized_meta_size))
                if not self.consider_objid and not self.consider_objsize:
                    sequences_train.append(Sequence(inter_times=seq))
            elif self.train_data_type == 'all':
                num_batches = int(np.ceil(len(meta.past_distances) / self.max_n_past_distances))
                for b in range(num_batches):
                    if b == num_batches - 1:
                        if self.use_pseudo_sample and pseudo_sample > 0 and not self.consider_survival:
                            seq = meta.past_distances[
                                  b * self.max_n_past_distances: (b + 1) * self.max_n_past_distances] + [
                                      pseudo_sample, self.current_seq - meta.past_timestamp]
                        else:
                            seq = meta.past_distances[b * self.max_n_past_distances: (b + 1) * self.max_n_past_distances] + [self.current_seq - meta.past_timestamp]
                    else:
                        seq = meta.past_distances[b * self.max_n_past_distances: (b + 1) * self.max_n_past_distances] + \
                                [np.random.uniform(meta.past_distances[(b + 1) * self.max_n_past_distances])]
                    if len(seq) == 1:
                        continue
                    normalized_meta_size = (meta.size - self.mean_obj_size) / self.std_obj_size
                    if self.consider_objid and not self.consider_objsize:
                        sequences_train.append(Sequence(inter_times=seq, obj_id=meta.mean_distance))
                    if not self.consider_objid and self.consider_objsize:
                        sequences_train.append(Sequence(inter_times=seq, obj_size=normalized_meta_size))
                    if self.consider_objid and self.consider_objsize:
                        sequences_train.append(Sequence(inter_times=seq, obj_id=meta.mean_distance, obj_size=normalized_meta_size))
                    if not self.consider_objid and not self.consider_objsize:
                        sequences_train.append(Sequence(inter_times=seq))

        logging.info("in_cache and out_cache metas contains %s objects. ",
                     len(self.in_cache_metas) + len(self.out_cache_metas))
        logging.info("training data contains %s sequences, cache contains %s objects",
                     len(sequences_train), len(self.cache_memory))

        dataset_train = SequenceDataset(sequences=sequences_train)
        d_train, d_val, d_test = dataset_train.train_val_test_split(seed=self.seed, train_size=0.8, val_size=0.1,
                                                                    test_size=0.1)
        d_val = d_val + d_test
        mean_log_inter_time, std_log_inter_time = d_train.get_inter_time_statistics()
        dl_train = d_train.get_dataloader(batch_size=self.ml_batch_size, shuffle=False)
        dl_val = d_val.get_dataloader(batch_size=self.ml_batch_size, shuffle=False)
        return dl_train, dl_val, mean_log_inter_time, std_log_inter_time

    def _train_process(self, model, dl_train, dl_val, opt, window_savename):
        impatient = 0
        best_loss = np.inf
        best_model = deepcopy(model.state_dict())
        training_val_losses = []
        for epoch in range(self.num_iterations):
            model.train()
            for batch in dl_train:
                opt.zero_grad()
                loss = -model.log_prob(batch).mean()
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                total_val_loss = 0.0
                total_val_count = 0
                for val_batch in dl_val:
                    total_val_loss += -model.log_prob(val_batch).sum()
                    total_val_count += val_batch.size
                loss_val = total_val_loss / total_val_count
                training_val_losses.append(loss_val)

            if (best_loss - loss_val) < 1e-4:
                impatient += 1
                if loss_val < best_loss:
                    best_loss = loss_val
                    best_model = deepcopy(model.state_dict())
            else:
                best_loss = loss_val
                best_model = deepcopy(model.state_dict())
                impatient = 0

            if impatient >= self.patience:
                break

            if epoch % self.display_step == 0:
                logging.info("Epoch %s: loss_train_last_batch = %s, loss_val = %s", epoch, loss.item(), loss_val)

        torch.save(best_model, window_savename)
        model.load_state_dict(best_model)

    def train(self):
        if self.window_index == -1:
            self.warm_index = self.current_seq

        if self.reload:
            self.window_index = ''
        else:
            self.window_index += 1
        training_start = time.time()
        logging.info("training tpp at %s", self.current_seq)
        dl_train, dl_val, mean_log_inter_time, std_log_inter_time= self._density_nn_dataset()

        # get model name
        window_savename = self.savename.format(window_index=self.window_index)

        # Define the model
        if self.learn_objective == 'inter':
            model = RecurrentTPPInter(
                mean_log_inter_time=mean_log_inter_time,
                std_log_inter_time=std_log_inter_time,
                context_size=self.context_size,
                rnn_type=self.rnn_type,
                num_features=self.num_features,
                consider_survival=self.consider_survival,
                num_mix_components=self.n_components,
                consider_objid=self.consider_objid,
                consider_objsize=self.consider_objsize
            )
        elif self.learn_objective == 'residual':
            model = RecurrentTPPResidual(
                mean_log_inter_time=mean_log_inter_time,
                std_log_inter_time=std_log_inter_time,
                context_size=self.context_size,
                rnn_type=self.rnn_type,
                num_features=self.num_features,
                consider_survival=self.consider_survival,
                num_mix_components=self.n_components,
                consider_objid=self.consider_objid,
                consider_objsize=self.consider_objsize
            )
        if self.load_ml:
            model.load_state_dict(torch.load(window_savename))
            self.ml_model = model
            return

        if self.reload:
            model.load_state_dict(torch.load(window_savename))
        opt = torch.optim.Adam(model.parameters(), weight_decay=self.regularization, lr=self.lr)

        # train
        self._train_process(model=model, dl_train=dl_train, dl_val=dl_val, opt=opt, window_savename=window_savename)
        self.ml_model = model
        logging.info("finished training ml model at %s took %s seconds", self.current_seq, time.time() - training_start)

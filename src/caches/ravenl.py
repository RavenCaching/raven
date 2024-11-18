import os
import logging
import time
import torch
import numpy as np
import collections
from copy import deepcopy
from collections import Counter
from os.path import join, splitext, basename, exists
from caches.lrb import LRBCache, TrainingData, Meta
from configparser import ConfigParser
from randomdict import RandomDict
from models.sequence import Sequence
from models.dataset import SequenceDataset
from models.batch import Batch
from models.tpp_inter import RecurrentTPPInter
from models.tpp_residual import RecurrentTPPResidual

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


class RavenLearn(LRBCache):
    def __init__(self, capacity, config: ConfigParser):
        super().__init__(capacity=capacity, config=config)
        # parameters
        # overwrite LRB
        self.sample_size = config.getint('ravenl', 'sample_size', fallback=64)
        self.batch_size = config.getint('ravenl', 'batch_size', fallback=131072)
        self.memory_window = config.getint('ravenl', 'memory_window', fallback=1000000)
        self.max_n_past_timestamps = config.getint('ravenl', 'max_n_past_timestamps', fallback=32)
        self.max_n_past_distances = self.max_n_past_timestamps - 1
        self.priority_size_function = config.get('ravenl', 'priority_size_function', fallback='identity')
        self.use_size = config.getboolean('ravenl', 'use_size', fallback=False)
        self.use_edc = config.getboolean('ravenl', 'use_edc', fallback=False)
        self.use_n_within = config.getboolean('ravenl', 'use_n_within', fallback=False)
        # Raven Learn unique
        self.learn_objective = config.get('ravenl', 'learn_objective', fallback='inter')
        self.dist_sample_size = config.getint('ravenl', 'dist_sample_size', fallback=10)
        self.history_type = config.get('ravenl', 'history_type', fallback='lrb')
        self.load_ml = config.getboolean('ravenl', 'load_ml', fallback=False)
        self.context_size = config.getint('ravenl', 'context_size', fallback=32)
        self.rnn_type = config.get('ravenl', 'rnn_type', fallback='GRU')
        self.ml_batch_size = config.getint('ravenl', 'ml_batch_size', fallback=256)
        self.reload = config.getboolean('ravenl', 'reload', fallback=False)
        self.savedir = config.get('ravenl', 'savedir', fallback='./ckpoints/{trace}_{tag}').format(trace=config.trace, tag=config.tag)
        self.savename = join(self.savedir, config.get('ravenl', 'savename', fallback='model_{window_index}'))
        if not exists(self.savedir):
            os.makedirs(self.savedir)
        self.consider_survival = config.getboolean('ravenl', 'consider_survival', fallback=True)
        self.cuda = config.getint('ravenl', 'cuda', fallback=0)
        torch.cuda.set_device(self.cuda)

        # ML learning
        self.num_features = 1  # Input feature to rnn
        self.n_components = 64
        self.regularization = 1e-5
        self.lr = 1e-3
        self.num_iterations = 1000
        self.display_step = 20
        self.patience = 50
        self.seed = 0

        # data structures
        self.window_index = -1
        self.warm_index = -1

        # load_ml and re_load cannot be True at the same time
        assert not (self.load_ml == True and self.reload == True)

    def rank(self):
        # if not trained yet, or in_cache_lru past memory window, use LRU
        candidate = next(iter(self.lru_cache))
        meta = self.in_cache_metas[candidate]
        if not self.ml_model or self.current_seq - meta.past_timestamp >= self.memory_window:
            return meta.id, self.cache_memory[meta.id]

        # sample objects and compute their future intervals
        pred_objs = self.sample(self.sample_size)
        pred_data = []
        pred_age = []
        pred_last_event_idx = []
        for pred_obj in pred_objs:
            pred_meta = self.in_cache_metas[pred_obj]
            dists = pred_meta.past_distances[-self.max_n_past_distances:]
            pred_last_event_idx.append(len(dists) - 1)
            dists =  dists + [0] * (self.max_n_past_distances - len(dists))
            pred_data.append(dists)
            pred_age.append(self.current_seq - pred_meta.past_timestamp)
        pred_data = np.array(pred_data)
        pred_data_batch = Batch(inter_times=torch.Tensor(pred_data), mask=torch.ones_like(torch.Tensor(pred_data)))

        if self.learn_objective == 'inter':
            pass
        elif self.learn_objective == 'residual':
            pred_age = torch.Tensor(np.array(pred_age))
        else:
            raise Exception('undefined learning objective %s' % self.learn_objective)

        with torch.no_grad():
            features = self.ml_model.get_features(pred_data_batch)
            context = self.ml_model.get_context(features, remove_last=False)  # (batch_size, seq_len, context_size)
            context = context[[idx for idx in range(len(pred_last_event_idx))], pred_last_event_idx, :]

            if self.learn_objective == 'residual':
                obj_ages = torch.log(pred_age + 1e-8)
                obj_ages = (obj_ages - self.ml_model.mean_log_inter_time) / self.ml_model.std_log_inter_time
                obj_ages = obj_ages.unsqueeze(-1)
                new_context = torch.cat([context, obj_ages], dim=-1)
                residual_time_dist = self.ml_model.get_residual_time_dist(new_context)
                residual_time_samples = residual_time_dist.sample((self.dist_sample_size,)).cpu().numpy()
            elif self.learn_objective == 'inter':
                inter_time_dist = self.ml_model.get_inter_time_dist(context)
                inter_time_samples = inter_time_dist.sample((1000,)).cpu().numpy()  # sample_size * object number
                residual_time_samples = []
                for obj_idx in range(len(pred_objs)):
                    valid_sample = inter_time_samples[:, obj_idx][inter_time_samples[:, obj_idx] > pred_age[obj_idx]]
                    if len(valid_sample) <= 0:
                        residual_time_samples.append(
                            np.array([(pred_age[obj_idx] + 2 * self.memory_window)] * self.dist_sample_size))
                    else:
                        if len(valid_sample) < self.dist_sample_size:
                            valid_sample = np.resize(valid_sample, (self.dist_sample_size,))
                        residual_time_samples.append((valid_sample[-self.dist_sample_size:] - pred_age[obj_idx]))
                residual_time_samples = np.array(residual_time_samples)

        if self.priority_size_function != 'identity':
            raise Exception("priority size function: %s not implemented yet!" % self.priority_size_function)
        if self.learn_objective == 'residual':
            max_residual_idx = np.argmax(residual_time_samples, axis=1)
        elif self.learn_objective == 'inter':
            max_residual_idx = np.argmax(residual_time_samples, axis=0)
        meta_idx = Counter(max_residual_idx).most_common(1)[0][0]
        dobj = pred_objs[meta_idx]
        return dobj, self.cache_memory[dobj]

    def _density_nn_dataset(self):
        sequences_train = []
        for seq in self.training_data.tpp_data:
            sequences_train.append(Sequence(inter_times=seq))
        logging.info("training data contains %s sequences, cache contains %s objects",
                     len(sequences_train), len(self.cache_memory))
        dataset_train = SequenceDataset(sequences=sequences_train)
        mean_log_inter_time, std_log_inter_time = dataset_train.get_inter_time_statistics()
        d_train, d_val, d_test = dataset_train.train_val_test_split(seed=self.seed, train_size=0.8, val_size=0.1,
                                                                    test_size=0.1)
        d_val = d_val + d_test
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
                num_mix_components=self.n_components
            )
        elif self.learn_objective == 'residual':
            model = RecurrentTPPResidual(
                mean_log_inter_time=mean_log_inter_time,
                std_log_inter_time=std_log_inter_time,
                context_size=self.context_size,
                rnn_type=self.rnn_type,
                num_features=self.num_features,
                consider_survival=self.consider_survival,
                num_mix_components=self.n_components
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

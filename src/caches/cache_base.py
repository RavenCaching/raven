import logging
from randomdict import RandomDict
import numpy as np


class Request(object):
    def __init__(self, seq, id, size, next_seq=None):
        self.seq = seq
        self.id = id
        self.size = size
        # Used for offline algorithms, such as Belady and BeladySize
        self.next_seq = int(next_seq) if next_seq is not None else None


class CacheStatistic(object):
    def __init__(self, policy, statistic_window=1000000):
        self.policy = policy
        self.statistic_window = statistic_window
        self.obj_hit = 0
        self.obj_total = 0
        self.byte_hit = 0
        self.byte_total = 0
        self.ohrs = []
        self.bhrs = []
        self.window_obj_hits = []
        self.window_byte_hits = []
        self.window_obj_requests = []
        self.window_byte_requests = []

    def update(self, hit, req: Request):
        self.obj_total += 1
        self.byte_total += req.size
        if hit:
            self.obj_hit += 1
            self.byte_hit += req.size
        if self.obj_total % self.statistic_window == 0:
            window_ohr = self.get_ohr()
            window_bhr = self.get_bhr()
            self.ohrs.append(window_ohr)
            self.bhrs.append(window_bhr)
            logging.info("policy: %s, total req %s, total bytes %s, ohr: %s, bhr: %s",
                         self.policy, self.obj_total, self.byte_total, window_ohr, window_bhr)
            self.window_obj_hits.append(self.obj_hit - np.sum(self.window_obj_hits))
            self.window_byte_hits.append(self.byte_hit - np.sum(self.window_byte_hits))
            self.window_obj_requests.append(self.obj_total - np.sum(self.window_obj_requests))
            self.window_byte_requests.append(self.byte_total - np.sum(self.window_byte_requests))

    def get_ohr(self):
        return float(self.obj_hit) / self.obj_total

    def get_bhr(self):
        return float(self.byte_hit) / self.byte_total

    def get_window_ohrs(self):
        return self.ohrs

    def get_window_bhrs(self):
        return self.bhrs

    def print(self, warm_index=-1):
        # last window
        if self.obj_total > self.statistic_window * len(self.window_obj_hits):
            window_ohr = self.get_ohr()
            window_bhr = self.get_bhr()
            self.ohrs.append(window_ohr)
            self.bhrs.append(window_bhr)
            logging.info("policy: %s, total req %s, total bytes %s, ohr: %s, bhr: %s",
                         self.policy, self.obj_total, self.byte_total, window_ohr, window_bhr)
            self.window_obj_hits.append(self.obj_hit - np.sum(self.window_obj_hits))
            self.window_byte_hits.append(self.byte_hit - np.sum(self.window_byte_hits))
            self.window_obj_requests.append(self.obj_total - np.sum(self.window_obj_requests))
            self.window_byte_requests.append(self.byte_total - np.sum(self.window_byte_requests))
        
        logging.info("window object hits: %s", self.window_obj_hits)
        logging.info("window object reqs: %s", self.window_obj_requests)
        logging.info("window byte hits: %s", self.window_byte_hits)
        logging.info("window byte reqs: %s", self.window_byte_requests)
        logging.info("window ohrs: %s", self.get_window_ohrs())
        logging.info("window bhrs: %s", self.get_window_bhrs())
        logging.info("policy: %s, total req %s, total bytes %s, ohr: %s, bhr: %s",
                     self.policy, self.obj_total, self.byte_total, self.get_ohr(), self.get_bhr())
        if warm_index > 0:
            warm_index = int(warm_index / self.statistic_window )
            no_warm_obj_hits = self.obj_hit - np.sum(self.window_obj_hits[0 : warm_index])
            no_warm_obj_reqs = self.obj_total - np.sum(self.window_obj_requests[0: warm_index])
            no_warm_ohr = float(no_warm_obj_hits) / no_warm_obj_reqs
            no_warm_byte_hits = self.byte_hit - np.sum(self.window_byte_hits[0: warm_index])
            no_warm_byte_reqs = self.byte_total - np.sum(self.window_byte_requests[0: warm_index])
            no_warm_bhr = float(no_warm_byte_hits) / no_warm_byte_reqs
            logging.info("policy: %s, NO warm-up reqs %s, bytes %s, ohr: %s, bhr: %s",
                         self.policy, no_warm_obj_reqs, no_warm_byte_reqs, no_warm_ohr, no_warm_bhr)

class CacheBase(object):
    def __init__(self, capacity):
        self.remain = capacity
        self.capacity = capacity
        # maps object to request
        self.cache_memory = RandomDict()

    def sample(self, sample_size=64):
        if len(self.cache_memory) <= sample_size:
            logging.debug("there are less or equal than %d objects in cache, no need to sample", sample_size)
            return [key for key in self.cache_memory]
        else:
            return self.cache_memory.sample_key(k=sample_size)

    def lookup(self, req: Request) -> bool:
        return req.id in self.cache_memory

    def admit(self, req: Request):
        if req.id in self.cache_memory:
            return
        self.remain -= req.size
        self.cache_memory[req.id] = req

    def evict(self, obj):
        self.remain += self.cache_memory[obj].size
        self.cache_memory.pop(obj)

    def print(self):
        logging.info("cache size: %s, remain size: %s, number of objects in cache: %s",
                     self.capacity, self.remain, len(self.cache_memory))

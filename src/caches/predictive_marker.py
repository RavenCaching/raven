# algorithm from Lykouris & Vassilvitsky (https://dblp.org/rec/conf/icml/LykourisV18.html)
# based on a ICML20 paper: https://github.com/adampolak/mts-with-predictions
import collections
import math
import random
from configparser import ConfigParser
from caches.cache_base import CacheBase, Request

class PredMarkerCache(CacheBase):
    def __init__(self, capacity, config: ConfigParser):
        super().__init__(capacity)
        self.uni_size = config.getboolean('common', 'uni_size', fallback=True)
        assert self.uni_size == True
        self.PRED_MARKER_GAMMA = 1.
        self.Hk = 1.
        for i in range(2, capacity + 1):
            self.Hk += 1 / i
        self.cache = [None] * capacity
        self.unmarked = list(range(capacity))
        self.cache_preds = [-1] * capacity
        self.clean_c = 0
        self.stale = {}
        self.chain_lengths = []
        self.chain_reps = []
        self.prev_occs = {}  # list of previous occurences of each element
        self.weights = []  # weights[i] = weight of the element requested i steps earlier (weights[0] -> current element)
        self.k = capacity
        self.sum_weights = 0

    def PredPLECO_BK(self, req):
        # Prediction PLECO tailored for the BK dataset (DOI: 10.1145/2566486.2568018)
        # Note: expensive computations, quadratic in len(requests)
        # the predictions we give (1 / probability(being the next element))
        t = req.seq + 1 # t starts at 1
        self.weights.append((t + 10) ** (-1.8) * math.exp(-t / 670))  # weights starts at t=1
        self.sum_weights += self.weights[-1]
        if req.id not in self.prev_occs:
            self.prev_occs[req.id] = []
        self.prev_occs[req.id].append(t)
        prob = sum(self.weights[t - i] for i in self.prev_occs[req.id]) / self.sum_weights  # probability that request is the next occurence according to PLECO: t-i in [0;t-1]
        pred = 1 / prob + t - 1  # predicted next occurence
        return pred

    def lookup(self, req: Request) -> bool:
        hit = super().lookup(req=req)
        if hit:
            index_to_evict = self.cache.index(req.id)
            self.cache[index_to_evict] = req.id
            # predicted next occurence
            pred = self.PredPLECO_BK(req=req)
            self.cache_preds[index_to_evict] = pred
            if index_to_evict in self.unmarked:
                self.unmarked.remove(index_to_evict)
        return hit

    def admit(self, req: Request):
        if None in self.cache:
            index_to_evict = self.cache.index(None)
        else:
            index_to_evict = self.evict(req)
        # admit the object
        super().admit(req)
        self.cache[index_to_evict] = req.id
        # predicted next occurence
        pred = self.PredPLECO_BK(req=req)
        self.cache_preds[index_to_evict] = pred
        if index_to_evict in self.unmarked:
            self.unmarked.remove(index_to_evict)

    def evict(self, req):
        if not self.unmarked:
            self.clean_c = 0
            self.stale = set(self.cache)
            self.unmarked = list(range(self.k))
            self.chain_lengths = []
            self.chain_reps = []
        if req.id not in self.stale:
            self.clean_c += 1
            index_to_evict = max((self.cache_preds[i], i) for i in self.unmarked)[1]
            self.chain_lengths.append(1)
            self.chain_reps.append(self.cache[index_to_evict])
        else:
            assert req.id in self.chain_reps
            c = self.chain_reps.index(req.id)
            self.chain_lengths[c] += 1
            if self.chain_lengths[c] <= self.Hk * self.PRED_MARKER_GAMMA:
                index_to_evict = max((self.cache_preds[i], i) for i in self.unmarked)[1]
            else:
                index_to_evict = random.choice(self.unmarked)
            self.chain_reps[c] = self.cache[index_to_evict]
        dobj_id = self.cache[index_to_evict]
        super().evict(obj=dobj_id)
        return index_to_evict

# a naive implementation of GDSF

import collections
from configparser import ConfigParser
from caches.cache_base import CacheBase, Request
from heapq import heapify, heappush, heappop


class GDSFCache(CacheBase):
    # Reference
    # https://www.geeksforgeeks.org/min-heap-in-python/
    # https://docs.python.org/3/library/heapq.html
    def __init__(self, capacity, config: ConfigParser):
        super().__init__(capacity)
        self.sample_size = config.getint('gdsf', 'sample_size', fallback=64)
        self.gds_flag = config.getboolean('gdsf', 'gds_flag', fallback=False)
        self.gdsf_cache = {}  # object_ID -> (frequency, rank_score)
        self.L = 0
        if self.sample_size <= 0:
            self.rank_heap = []  # (rank_score, object_ID)
            heapify(self.rank_heap)

    def lookup(self, req: Request) -> bool:
        hit = super().lookup(req=req)
        # update and maintain the order
        if hit:
            new_freq = self.gdsf_cache[req.id][0] +  1
            if self.gds_flag:
                new_rank = 1.0 / req.size + self.L
            else:
                new_rank = float(new_freq) / req.size + self.L
            self.gdsf_cache[req.id] = (new_freq, new_rank)
            if self.sample_size <= 0:
                heappush(self.rank_heap, (self.gdsf_cache[req.id][1], req.id))
        return hit

    def admit(self, req: Request):
        # admit the object
        super().admit(req)
        # maintain the order
        self.gdsf_cache[req.id] = (1, 1.0/req.size + self.L)
        if self.sample_size <= 0:
            heappush(self.rank_heap, (self.gdsf_cache[req.id][1], req.id))
        while self.remain < 0:
            self.evict(None)

    def evict(self, obj):
        if self.sample_size <= 0:
            # heap has duplicated object meta data
            while True:
                dobj_rank, dobj_id = heappop(self.rank_heap)
                if dobj_id in self.gdsf_cache:
                    if self.gdsf_cache.get(dobj_id)[1] == dobj_rank:
                        break
            self.L = dobj_rank
            self.gdsf_cache.pop(dobj_id)
            super().evict(obj=dobj_id)
        else:
            dobjs = self.sample(self.sample_size)
            dobj_id, (dobj_freq, dobj_rank) = min([(dobj, self.gdsf_cache[dobj]) for dobj in dobjs], key=lambda k: k[1][1])
            self.L = dobj_rank
            self.gdsf_cache.pop(dobj_id)
            super().evict(obj=dobj_id)

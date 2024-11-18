import collections
from configparser import ConfigParser
from caches.cache_base import CacheBase, Request


class LRUCache(CacheBase):
    # Reference
    # https://www.geeksforgeeks.org/lru-cache-in-python-using-ordereddict/
    def __init__(self, capacity, config: ConfigParser):
        super().__init__(capacity)
        self.sample_size = config.getint('lru', 'sample_size', fallback=-1)
        if self.sample_size <= 0:
            self.lru_cache = collections.OrderedDict()

    def lookup(self, req: Request) -> bool:
        hit = super().lookup(req=req)
        if self.sample_size <= 0:
            # maintain the order
            if hit:
                self.lru_cache.move_to_end(req.id)
        return hit

    def admit(self, req: Request):
        if self.sample_size <= 0:
            # admit the object
            super().admit(req)
            # maintain the order
            self.lru_cache[req.id] = req
            self.lru_cache.move_to_end(req.id)
            while self.remain < 0:
                self.evict(None)
        else:
            # admit the object
            super().admit(req)
            while self.remain < 0:
                self.evict(None)

    def evict(self, obj):
        if self.sample_size <= 0:
            dobj, dreq = self.lru_cache.popitem(last=False)
            super().evict(obj=dobj)
        else:
            dobjs = self.sample(self.sample_size)
            dreq = min([self.cache_memory[dobj] for dobj in dobjs], key=lambda k: k.seq)
            super().evict(obj=dreq.id)

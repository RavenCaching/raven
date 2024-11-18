from os.path import exists
from configparser import ConfigParser
from sortedcontainers import SortedDict
from caches.cache_base import CacheBase, Request


max_next_seq = 0xffffffff


def annotate(trace_file):
    atrace_file = trace_file + ".ant"
    if exists(atrace_file):
        print("reusing cached annotated file: %s" % atrace_file)
        return atrace_file

    id_and_next_seq = []
    with open(trace_file) as inf:
        for line_index, line in enumerate(inf):
            obj_time, obj_id, obj_size = line.split(' ')
            if line_index % 1000000 == 0:
                print("reading origin trace: %s" % line_index)
            id_and_next_seq.append(obj_id)
    n_req = len(id_and_next_seq)
    print("scanned trace n=%s" % n_req)
    last_seen = {}
    for req_id in range(n_req-1, -1, -1):
        obj_id = id_and_next_seq[req_id]
        if obj_id in last_seen:
            id_and_next_seq[req_id] = last_seen[obj_id]
        else:
            id_and_next_seq[req_id] = max_next_seq
        last_seen[obj_id] = req_id
        if req_id % 1000000 == 0:
            print("computing next t: %s" % req_id)

    # write to the annotated trace file
    with open(atrace_file, 'w') as outf:
        with open(trace_file) as inf:
            for line_index, line in enumerate(inf):
                req_id = line_index
                outf.write('%s %s' % (id_and_next_seq[req_id], line))
                if line_index % 1000000 == 0:
                    print("writing: %s" % line_index)
    print("wrote trace n=%s" % n_req)
    return atrace_file


class Belady(CacheBase):
    def __init__(self, capacity, config: ConfigParser):
        super().__init__(capacity=capacity)
        self.sample_size = config.getint('belady', 'sample_size', fallback=-1)
        self.always_admission = config.getboolean('belady', 'always_admission', fallback=True)
        if self.sample_size <= 0:
            # maps next request sequence to object id
            self._next_req_map = SortedDict()

    def lookup(self, req: Request) -> bool:
        if self.sample_size <= 0:
            self._next_req_map.setdefault(req.next_seq, set())
            self._next_req_map[req.next_seq].add(req.id)
            self._next_req_map.pop(req.seq, None)
        return super().lookup(req=req)

    def admit(self, req: Request):
        # if not always admission, don't admit objects that will never come again
        if not self.always_admission and req.next_seq == max_next_seq:
            if self.sample_size <= 0:
                self._next_req_map[max_next_seq].remove(req.id)
                if len(self._next_req_map[max_next_seq]) == 0:
                    self._next_req_map.pop(max_next_seq)
            return
        # admit the object
        super().admit(req)
        # always update the request, the parent class keeps the earliest request
        self.cache_memory[req.id] = req
        # evict objects
        while self.remain < 0:
            self.evict(None)

    def evict(self, obj):
        if self.sample_size <= 0:
            dnext_req = self._next_req_map.iloc[-1]
            dobj = self._next_req_map[dnext_req].pop()
            if len(self._next_req_map[dnext_req]) == 0:
                self._next_req_map.pop(dnext_req)
            super().evict(obj=dobj)
        else:
            dobjs = self.sample(self.sample_size)
            dreq = max([self.cache_memory[dobj] for dobj in dobjs], key=lambda k: k.next_seq)
            super().evict(obj=dreq.id)


class BeladySize(CacheBase):
    def __init__(self, capacity, config: ConfigParser):
        super().__init__(capacity=capacity)
        self.sample_size = config.getint('beladysize', 'sample_size', fallback=-1)
        self.furthest_size = config.getint('beladysize', 'furthest_size', fallback=64)
        self.always_admission = config.getboolean('beladysize', 'always_admission', fallback=True)
        self.current_seq = -1
        if self.sample_size <= 0:
            # maps next request sequence to object id
            self._next_req_map = SortedDict()

    def lookup(self, req: Request) -> bool:
        self.current_seq += 1
        if self.sample_size <= 0:
            self._next_req_map.setdefault(req.next_seq, set())
            self._next_req_map[req.next_seq].add(req.id)
            self._next_req_map.pop(req.seq, None)
        return super().lookup(req=req)

    def admit(self, req: Request):
        # if not always admission, don't admit objects that will never come again
        if not self.always_admission and req.next_seq == max_next_seq:
            if self.sample_size <= 0:
                self._next_req_map[max_next_seq].remove(req.id)
                if len(self._next_req_map[max_next_seq]) == 0:
                    self._next_req_map.pop(max_next_seq)
            return
        # admit the object
        super().admit(req)
        # always update the request, the parent class keeps the earliest request
        self.cache_memory[req.id] = req
        # evict objects
        while self.remain < 0:
            self.evict(None)

    def evict(self, obj):
        if self.sample_size <= 0:
            dnext_reqs = self._next_req_map.iloc[-self.furthest_size:]
            dobjs = {}
            for dnext_req in dnext_reqs:
                for obj_id in self._next_req_map[dnext_req]:
                    dobjs[obj_id] = dnext_req
            dobj = max(dobjs, key=lambda k: (dobjs[k] - self.current_seq) * self.cache_memory[k].size)
            dnext_req = dobjs[dobj]
            if len(self._next_req_map[dnext_req]) <= 1:
                # shouldn't be less than 1
                self._next_req_map.pop(dnext_req)
            else:
                self._next_req_map[dnext_req].remove(dobj)
            super().evict(obj=dobj)
        else:
            dobjs = self.sample(self.sample_size)
            dreq = max([self.cache_memory[dobj] for dobj in dobjs], key=lambda k: (k.next_seq - self.current_seq) * k.size)
            super().evict(obj=dreq.id)

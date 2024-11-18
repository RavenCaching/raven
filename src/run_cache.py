import os
import time
import logging
import argparse
from configparser import ConfigParser
from os.path import join, splitext, basename, exists
from caches.cache_base import Request, CacheStatistic
from caches.lru import LRUCache
from caches.lrb import LRBCache
from caches.ravenh import RavenHeuristic
from caches.ravenl import RavenLearn
from caches.ravenltao import RavenLearnTao
from caches.raven_ensemble import RavenLearnEnsemble
from caches.raven_truncated import RavenLearnTruncated
from caches.raven_lrb import RavenLRB
from caches.belady import annotate, Belady, BeladySize
from caches.GDSF import GDSFCache
from caches.predictive_marker import PredMarkerCache


def main():
    parser = argparse.ArgumentParser(description='Run specified caching algorithms')
    parser.add_argument('trace_file', type=str, help='Path to the cache trace file')
    parser.add_argument('cache_size', type=int, help='The size of cache')
    parser.add_argument('--logfile', help='Path to the log file')
    parser.add_argument('-t', '--logtag', default='', help='The log tag to add in the log filename')
    parser.add_argument('-c', '--config_file', default='./policy.ini', dest='config_file',
                        help='Path to the config file')
    parser.add_argument('-p', '--cache_policies', dest='cache_policies', nargs='+',
                        choices=['lru', 'lrb', 'ravenh', 'ravenl', 'ravenltao', 'ravenlensemble',
                                 'ravenltruncated', 'ravenlrb', 'gdsf', 'belady', 'beladysize', 'predmarker'], required=True,
                        help='The cache policies to run')

    args = parser.parse_args()
    config = ConfigParser()
    if args.config_file:
        config.read_file(open(args.config_file))
    if args.logfile:
        logging.basicConfig(filename=args.logfile, level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        logging_misses = open(args.logfile + '_misses.txt', 'w')
    else:
        logdir = config.get('common', 'logdir', fallback='./log')
        if not exists(logdir):
            os.makedirs(logdir)
        logname = config.get('common', 'logname', fallback='{policy}.log')
        logname = logname.format(
            trace=splitext(basename(args.trace_file))[0],
            policy='-'.join(args.cache_policies),
            capacity=args.cache_size,
            tag=args.logtag)
        logging.basicConfig(filename=join(logdir, logname), level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
        logging_misses = open(join(logdir, logname + '_misses.txt'), 'w')


    # open and load cache streaming request
    simulation_start = time.time()
    time_format = config.get('common', 'time_format', fallback='seq_num')
    statistic_window = config.getint('common', 'statistic_window', fallback=1000000)
    early_stop = config.getint('common', 'early_stop', fallback=-1)
    uni_size = config.getboolean('common', 'uni_size', fallback=False)
    runtime_log = config.getboolean('common', 'runtime_log', fallback=False)

    caches = []
    cache_stats = []
    is_offline = 'belady' in args.cache_policies or 'beladysize' in args.cache_policies
    if is_offline:
        read_trace_file = annotate(trace_file=args.trace_file)
    else:
        read_trace_file = args.trace_file

    for policy in args.cache_policies:
        if policy == 'lru':
            caches.append(LRUCache(capacity=args.cache_size, config=config))
        elif policy == 'gdsf':
            caches.append(GDSFCache(capacity=args.cache_size, config=config))
        elif policy == 'predmarker':
            caches.append(PredMarkerCache(capacity=args.cache_size, config=config))
        elif policy == 'lrb':
            caches.append(LRBCache(capacity=args.cache_size, config=config))
        elif policy == 'ravenh':
            caches.append(RavenHeuristic(capacity=args.cache_size, config=config))
        elif policy == 'ravenl':
            config.trace = splitext(basename(args.trace_file))[0]
            config.tag = args.logtag
            caches.append(RavenLearn(capacity=args.cache_size, config=config))
        elif policy == 'ravenltao':
            config.trace = splitext(basename(args.trace_file))[0]
            config.tag = args.logtag
            caches.append(RavenLearnTao(capacity=args.cache_size, config=config))
        elif policy == 'ravenlensemble':
            config.trace = splitext(basename(args.trace_file))[0]
            config.tag = args.logtag
            caches.append(RavenLearnEnsemble(capacity=args.cache_size, config=config))
        elif policy == 'ravenltruncated':
            config.trace = splitext(basename(args.trace_file))[0]
            config.tag = args.logtag
            caches.append(RavenLearnTruncated(capacity=args.cache_size, config=config))
        elif policy == 'ravenlrb':
            config.trace = splitext(basename(args.trace_file))[0]
            config.tag = args.logtag
            caches.append(RavenLRB(capacity=args.cache_size, config=config))
        elif policy == 'belady':
            caches.append(Belady(capacity=args.cache_size, config=config))
        elif policy == 'beladysize':
            caches.append(BeladySize(capacity=args.cache_size, config=config))
        else:
            raise Exception("Unexpected cache policy: %s" % policy)
        cache_stats.append(CacheStatistic(policy=policy, statistic_window=statistic_window))

    logging.info("Caching simulation starts")
    stop_flag = False
    for line_count, line in enumerate(open(read_trace_file)):
        if is_offline:
            next_seq, obj_request_time, obj_id, obj_size = line.strip().split(' ')
        else:
            next_seq = None
            obj_request_time, obj_id, obj_size = line.strip().split(' ')
        if uni_size:
            obj_size = 1
        if time_format == 'seq_num':
            req = Request(seq=line_count, id=obj_id, size=float(obj_size), next_seq=next_seq)
        elif time_format == 'absolute':
            # FIXME: next_seq is generated using using seq_num. so absolute doesn't apply
            req = Request(seq=float(obj_request_time), id=obj_id, size=float(obj_size))
        else:
            raise Exception("Unexpected time format option: %s" % time_format)
        if early_stop > 0 and line_count >= early_stop:
            break
        for cache_index, cache in enumerate(caches):
            if runtime_log and (cache.admitting_count >= 100 and cache.evicting_count >= 100):
                logging.info("ML training time: %s, average admit time: %s, average evict time: %s",
                             cache.ml_training_time, cache.admitting_time / cache.admitting_count,
                             cache.evicting_time / cache.evicting_count)
                stop_flag = True
            # look up the object
            hit = cache.lookup(req)
            cache_stats[cache_index].update(hit=hit, req=req)
            if not hit:
                logging_misses.write(str(line_count) + ',' + str(obj_id) + '\n')
                # admit the object
                cache.admit(req=req)
        if runtime_log and stop_flag:
            break
    logging.info("Caching simulation ends, time elapsed: %s", time.time() - simulation_start)
    logging.info("Final caching results:")
    for cache_index, cache in enumerate(caches):
        warm_index = cache.warm_index if hasattr(cache, 'warm_index') else -1
        cache_stats[cache_index].print(warm_index=warm_index)
        n_force_eviction = cache.n_force_eviction if hasattr(cache, 'n_force_eviction') else -1
        n_random_guess = cache.n_random_guess if hasattr(cache, 'n_random_guess') else -1
        logging.info("n_force_eviction = %s, n_random_guess = %s", n_force_eviction, n_random_guess)
        sole_tpp_num = cache.sole_tpp_num if hasattr(cache, 'sole_tpp_num') else -1
        sole_gbm_num = cache.sole_gbm_num if hasattr(cache, 'sole_gbm_num') else -1
        comb_tpp_gbm = cache.comb_tpp_gbm if hasattr(cache, 'comb_tpp_gbm') else -1
        rank_num = cache.rank_num if hasattr(cache, 'rank_num') else -1
        logging.info("sole_tpp_num = %s, sole_gbm_num = %s, comb_tpp_gbm = %s, rank_num = %s", sole_tpp_num, sole_gbm_num, comb_tpp_gbm, rank_num)

if __name__=="__main__":
    main()

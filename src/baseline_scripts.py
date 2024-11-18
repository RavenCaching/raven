import ast
import logging
import argparse
import subprocess
from os.path import join, dirname, basename, abspath


def run_pfool(trace, cache_size, pfool_path='../optimalwebcaching/BHRgoal/PFOO-L/pfool'):
    print('PFOOL')
    pfool_cmd = [pfool_path, trace, str(cache_size)]
    stdout = subprocess.check_output(pfool_cmd)
    results = stdout.decode().strip().split(' ')
    ohr = results[-3]
    bhr = results[-1]
    logging.warning("PFOOL -- ohr: %s, bhr: %s", ohr, bhr)
    return ohr, bhr


def run_belady(trace, cache_size, belady_path='../optimalwebcaching/BHRgoal/Belady/belady2', sample_rate=64):
    print('belady')
    run_cmd = [belady_path, trace, str(cache_size), str(sample_rate)]
    stdout = subprocess.check_output(run_cmd)
    results = stdout.decode().strip().split(' ')
    ohr = results[-3]
    bhr = results[-1]
    logging.warning("belady -- ohr: %s, bhr: %s", ohr, bhr)
    return ohr, bhr


def run_belady_size(trace, cache_size, run_path='../optimalwebcaching/OHRgoal/Belady-Size/belady2size', sample_rate=64):
    print('belady2size')
    run_cmd = [run_path, trace, str(cache_size), str(sample_rate)]
    stdout = subprocess.check_output(run_cmd)
    results = stdout.decode().strip().split('\n')
    ohr = results[0].split(' ')[-1]
    bhr = results[1].split(' ')[-1]
    logging.warning("belady2size -- ohr: %s, bhr: %s", ohr, bhr)
    return ohr, bhr


def run_lhr(trace, cache_size, run_path='../lhr-work/src/Main.py'):
    print('LHR')
    run_cmd = ['python3', run_path, trace, str(cache_size)]
    stdout = subprocess.check_output(run_cmd)
    ohr = float(stdout.decode().strip())
    logging.warning("LHR -- ohr: %s", ohr)
    return ohr, None


# algorithm = 'LRB', 'LHD', 'Hyperbolic'
# memory_window=671088
def run_lrb(trace, cache_size, algorithm, memory_window=10000000, bloom_filter=0, sample_rate=64,
            max_n_past_timestamps=32, n_edc_feature=10, objective="byte_miss_ratio", no_sizeFeature=0,
            no_nwithin=0, no_LRU=0, log_start_seq=-1, batch_size=131072, uni_size=False,
            result_filename='NoNeed', n_early_stop=-1, use_seq_to_train=0):
    trace_path = abspath(trace)
    print(algorithm)
    if uni_size:
        uni_size = 1
    else:
        uni_size = 0
    lrb_cmd = ['sudo', 'docker', 'run', '-it', 
              '-v', '%s/lrb_log/%s/:/tmp/' % (dirname(trace_path), result_filename.strip('.csv')),
              '-v', '%s:/trace' % dirname(trace_path),
              'sunnyszy/webcachesim:0.7New',
               basename(trace_path), algorithm, str(cache_size), '--memory_window=%s' % memory_window,
               '--bloom_filter=%s' % bloom_filter, '--sample_rate=%s' % sample_rate,
               '--max_n_past_timestamps=%s' % max_n_past_timestamps, '--n_edc_feature=%s' % n_edc_feature,
               '--objective=%s' % objective, '--no_sizeFeature=%s' % no_sizeFeature, '--no_nwithin=%s' % no_nwithin,
               '--no_LRU=%s' % no_LRU, '--log_start_seq=%s' % log_start_seq, '--batch_size=%s' % batch_size,
               '--uni_size=%s' % uni_size, '--n_early_stop=%s' % n_early_stop, '--use_seq_to_train=%s' % use_seq_to_train]
    stdout = subprocess.check_output(lrb_cmd)
    summary = [line for line in stdout.decode('utf-8', 'ignore').split('\n') if 'no_warmup_byte_miss_ratio' in line][0]
    summary_dict = ast.literal_eval(summary)
    if algorithm == 'LRB':
        print('segment_n_retrain:', summary_dict['segment_n_retrain'])

    warm_idx = int(batch_size / 1000000)
    print('warm up index: ', warm_idx)
    print("segment_object_miss: ", summary_dict['segment_object_miss'])
    print("segment_object_req: ", summary_dict['segment_object_req'])
    print("segment_byte_miss: ", summary_dict['segment_byte_miss'])
    print("segment_byte_req: ", summary_dict['segment_byte_req'])

    no_warmup_bhr = 1 - sum(summary_dict['segment_byte_miss'][warm_idx:]) / sum(summary_dict['segment_byte_req'][warm_idx:])
    no_warmup_ohr = 1 - sum(summary_dict['segment_object_miss'][warm_idx:]) / sum(summary_dict['segment_object_req'][warm_idx:])
    average_segment_obj_miss_ratio = sum(summary_dict['segment_object_miss']) / sum(summary_dict['segment_object_req'])
    average_segment_byte_miss_ratio = sum(summary_dict['segment_byte_miss']) / sum(summary_dict['segment_byte_req'])
    logging.warning("algorithm %s, no_warmup_ohr: %s, no_warmup_bhr: %s", algorithm, no_warmup_ohr, no_warmup_bhr)
    if use_seq_to_train > 0:
        return no_warmup_ohr, no_warmup_bhr
    else:
        return 1 - average_segment_obj_miss_ratio, 1 - average_segment_byte_miss_ratio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('trace', help='Path to the trace file')
    parser.add_argument('cache_size', type=int, help='The size of the cache in bytes')
    parser.add_argument('--algorithms', nargs='+',
                        choices=['all', 'GDSF','pfool', 'lrb', 'foo', 'LHD', 'UCB', 'belady', 'belady2size', 'LRU',
                                 'LFUDA', 'AdaptSize', 'Hyperbolic', 'ThS4LRU', 'LHR'],
                        required=True, help='List of algorithms to run')
    parser.add_argument('--uni_size', default=False, action='store_true', help='whether to consider object size ')
    parser.add_argument('--bloom_filter', type=int, default=0, help='1 to use bloom_filter')
    parser.add_argument('--memory_window', type=int, default=10000000)
    parser.add_argument('--sample_rate', type=int, default=64)
    parser.add_argument('--max_n_past_timestamps', type=int, default=32)
    parser.add_argument('--n_edc_feature', type=int, default=10)
    parser.add_argument('--objective', type=str, default="byte_miss_ratio", help='byte_miss_ratio or object_miss_ratio')
    parser.add_argument('--no_sizeFeature', type=int, default=0)
    parser.add_argument('--no_nwithin', type=int, default=0)
    parser.add_argument('--no_LRU', type=int, default=0, help='when ML is available, 0 means use LRU eviction first; 1 means not use LRU eviction ')
    parser.add_argument('--batch_size', type=int, default=131072, help='when to train gbdt')
    parser.add_argument('--log_start_seq', type=int, default=-1, help='from what seq num to log info, -1 means not log any info, ')
    parser.add_argument('--n_early_stop', type=int, default=-1)
    parser.add_argument('--use_seq_to_train', type=int, default=0)


    args = parser.parse_args()

    result_filename = args.trace.split('/')[-1].split('.')[0] + '_baselineResults' + '_policy'+ '-'.join(args.algorithms) + \
                      'C' + str(args.cache_size) + '_win' + str(args.memory_window)
    if args.sample_rate != 64:
        result_filename = result_filename + '_sample' + str(args.sample_rate)
    if args.algorithms == 'lrb':
        if args.max_n_past_timestamps != 32:
            result_filename = result_filename + '_past' + str(args.max_n_past_timestamps)
        if args.n_edc_feature != 10:
            result_filename = result_filename + '_edc' + str(args.n_edc_feature)
        result_filename = result_filename + args.objective
        if args.no_sizeFeature == 1:
            result_filename = result_filename + '_noSizeFeature'
        if args.no_nwithin == 1:
            result_filename = result_filename + '_noNwithinFeature'
        if args.no_LRU == 1:
            result_filename = result_filename + '_noLRU'
        if args.batch_size != 131072:
            result_filename = result_filename + '_batch' + str(args.batch_size)
        if args.uni_size:
            result_filename = result_filename + '_uniSize'
        if args.n_early_stop != -1:
            result_filename = result_filename + '_stop' + str(args.n_early_stop)
        if args.use_seq_to_train != 0:
            result_filename = result_filename + '_seqTrain'
    result_filename = result_filename + '.csv'
    print(result_filename)
    results_file = open(join('./log', result_filename), 'w')

    if 'all' in args.algorithms:
        belady2size_ohr, belady2size_bhr = run_belady_size(trace=args.trace, cache_size=args.cache_size,
                                                           sample_rate=args.sample_rate)
        results_file.write('Belady-size,' + str(belady2size_ohr) + ',' + str(belady2size_bhr) + '\n')
        baselines = ['Belady', 'LRU', 'LFUDA', 'Hyperbolic', 'AdaptSize', 'LRB', 'LHD']
        for base_policy in baselines:
            ohr, bhr = run_lrb(trace=args.trace, cache_size=args.cache_size, algorithm=args.algorithms,
                                        bloom_filter=args.bloom_filter, memory_window=args.memory_window,
                                        sample_rate=args.sample_rate, max_n_past_timestamps=args.max_n_past_timestamps,
                                        n_edc_feature=args.n_edc_feature, objective=args.objective,
                                        no_sizeFeature=args.no_sizeFeature, no_nwithin=args.no_nwithin,
                                        no_LRU=args.no_LRU, log_start_seq=args.log_start_seq,
                                        batch_size=args.batch_size,
                                        result_filename=result_filename, uni_size=args.uni_size,
                                        n_early_stop=args.n_early_stop, use_seq_to_train=args.use_seq_to_train)
            results_file.write(base_policy + ',' + str(ohr) + ',' + str(bhr) + '\n')

    if 'pfool' in args.algorithms:
        pfool_ohr, pfool_bhr = run_pfool(trace=args.trace, cache_size=args.cache_size)
        results_file.write('PFOO,'+str(pfool_ohr)+','+str(pfool_bhr)+'\n')
    elif 'belady2size' in args.algorithms:
        belady2size_ohr, belady2size_bhr = run_belady_size(trace=args.trace, cache_size=args.cache_size, sample_rate=args.sample_rate)
        results_file.write('Belady-size,' + str(belady2size_ohr) + ',' + str(belady2size_bhr) + '\n')
    elif args.algorithms in ['Belady', 'LRU', 'LFUDA', 'Hyperbolic', 'AdaptSize', 'LRB', 'LHD']:
        ohr, bhr = run_lrb(trace=args.trace, cache_size=args.cache_size, algorithm=args.algorithms,
                                        bloom_filter=args.bloom_filter, memory_window=args.memory_window,
                                        sample_rate=args.sample_rate, max_n_past_timestamps=args.max_n_past_timestamps,
                                        n_edc_feature=args.n_edc_feature, objective=args.objective,
                                        no_sizeFeature=args.no_sizeFeature, no_nwithin=args.no_nwithin,
                                        no_LRU=args.no_LRU, log_start_seq=args.log_start_seq,
                                        batch_size=args.batch_size,
                                        result_filename=result_filename, uni_size=args.uni_size,
                                        n_early_stop=args.n_early_stop, use_seq_to_train=args.use_seq_to_train)
        results_file.write(args.algorithms + ',' + str(ohr) + ',' + str(bhr) + '\n')

    if 'LHR' in args.algorithms or 'lhr' in args.algorithms:
        ohr, bhr = run_lhr(trace=args.trace, cache_size=args.cache_size)
        results_file.write('LHR,' + str(ohr) + ',' + str(bhr) + '\n')

if __name__ == "__main__":
    #python3.6 baseline_scripts.py ../traces/wiki2018_abnormal_2T_downscale1000_new3c.tr 8388608 --algorithms all
    main()

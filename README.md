# Raven: Python Caching Simulator
A python re-implementation of c++ based [webcachesim](https://github.com/dasebe/webcachesim) and [webcachesim2](https://github.com/sunnyszy/lrb). It simulates a variety of existing caching policies by replaying request traces, and use this framework as a basis for `fast and interactive` experiment with new ones.

This simulator was built to evaluate [Raven: Belady-Guided, Predictive (Deep) Learning for In-Memory and Content Caching](https://dl.acm.org/doi/abs/10.1145/3555050.3569134) 
 (ACM CoNEXT, 2022).

Currently supported caching algorithms:
- Raven
- Learning Relaxed Belady (LRB)
- LRU
- Belady, Belady-sample (a sample-based approximate version), Belady-size (object size aware)
- GDSF
- PredMarker ([Online metric algorithms with untrusted predictions](https://github.com/adampolak/mts-with-predictions))

Configuration parameters of these algorithms can be found in the config file [src/policy.ini](https://github.com/RavenCaching/raven/blob/main/src/policy.ini).

## Install dependency
```
pip install bloom_filter, pytorch
```

## Request traces
- We provide a sample of wiki2018 trace. 
- The full wiki and twitter request traces can be downloaded from our google drive: [wiki2018](https://drive.google.com/file/d/1LxKznCejOl_aFhTsC99yBIYM2kwhjdDt/view?usp=drive_link), 
[wiki2019](https://drive.google.com/file/d/1nFgo0MYoop6g87kE_4YIByu1mKC3JLdm/view?usp=drive_link), [twitter17](https://drive.google.com/file/d/1AqkivM-ju3NtXMbw8SaSMAoWTu61elMt/view?usp=drive_link),
[twitter29](https://drive.google.com/file/d/143JgWKlY9syrI05DdVHhdUcZIeVwPBVq/view?usp=drive_link), [twitter45](https://drive.google.com/file/d/1hmgyShbLs8EYwWqJtf3Yo_DKjDOJ0P03/view?usp=drive_link),
[twitter52](https://drive.google.com/file/d/16GQqkWB8Ul1cXjKkkYKRP_NAqXdSPb4-/view?usp=drive_link).

- The `citi` traces are CitiBike data used by MTS([Online metric algorithms with untrusted predictions](https://github.com/adampolak/mts-with-predictions)).

## Trace Format
Request traces are expected to be in a space-separated format with 3 columns:
- time should be a long long int, but can be arbitrary (for future TTL feature, not currently in use)
- id should be a long long int, used to uniquely identify objects
- size should be uint32, this is object's size in bytes

## Test commands
The basic interface is
 ```
 python3 run_cache.py $trace $cache_size -p $cache_policy [--param=value]
 ```
Two Mixture Density Network pre-trained models can be accessed in [src/ckpoints](https://github.com/RavenCaching/raven/tree/main/src/ckpoints).

The [src/baseline_scripts.py](https://github.com/RavenCaching/raven/blob/main/src/baseline_scripts.py) contains python wrapper to quickly run all caching policies in [webcachesim](https://github.com/dasebe/webcachesim) and [webcachesim2](https://github.com/sunnyszy/lrb).

If you find this work useful for your research, please cite:
```bibtext
@inproceedings{hu2022raven,
  title={Raven: belady-guided, predictive (deep) learning for in-memory and content caching},
  author={Hu, Xinyue and Ramadan, Eman and Ye, Wei and Tian, Feng and Zhang, Zhi-Li},
  booktitle={Proceedings of the 18th International Conference on emerging Networking EXperiments and Technologies},
  pages={72--90},
  year={2022}
}

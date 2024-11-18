# install dependency
1) install ifl-tpp:

    git clone https://github.com/shchur/ifl-tpp.git
    
    python3.6 setup.py install
    
2) install bloom-filter

    pip3.6 install bloom_filter


# baseline: 
  1) cache-policy with object size: adaptsize, approximate-belady, OPT-size(offline, oracle algorithm) (TODO)
  2) cache-policy without object size: caffine simulator.
  
  
# implementation framework
1. data preprocessing, analyze the inter-arrival coorelation of different objects (in real trace dataset).
2. integrating size
  

# test commands

- on small synthetic trace
    ```python3 main.py D1b_O1000_Pareto_converted.tr 12878 --policy=CDF_residual_age_learn --use_obj_size --priority_size_function=linear --load_ml```
    
- on large real trace
    ```python3 main.py wiki2018_downscale10_head100M_converted.tr 134217728 --policy=CDF_residual_age_learn --use_obj_size --priority_size_function=linear --load_ml```


# validation commands
- for LRU
    ```
    python3 run_cache.py ../traces/cluster29_raven_lrb_downscale10_downscale20.tr 262144 -p lru
    0.6338574479	0.6948330819
    python3 run_cache.py ../traces/cluster29_raven_lrb_downscale10_downscale20_uniSize_lowFreq25.tr 50000 -p lru
    0.6102240524
    python3 run_cache.py ../traces/wiki2019_remapped_downscale10_downscale20_webcachesim_uniSize.tr 4000 -p lru
    0.5531405511
    python3 run_cache.py ../traces/wiki2019_remapped_downscale10_downscale20_webcachesim.tr 134217728 -p lru
    0.4626562875	0.2582136834
    ```
- for LRB
    ```
    python3 run_cache.py ../traces/cluster29_raven_lrb_downscale10_downscale20.tr 262144 -p lrb
    LRB 0.6716869049	0.7385436622
    LRB 0.6705450985	0.7376577813
    python3 run_cache.py ../traces/cluster45_raven_lrb_downscale20.tr 8388608 -p lrb
    LRB	0.5264895663	0.6652467161
    python3 run_cache.py ../traces/cluster17_raven_lrb_downscale10_downscale30.tr 8192 -p lrb 
    LRB 0.7855810905	0.8093617575
    python3 run_cache.py ../traces/wiki2019_remapped_downscale10_downscale20_webcachesim.tr 2147483648 -p lrb
    LRB 0.9189164635	0.7584358409
    python3 run_cache.py ../traces/wiki2018_downscale100_webcachesim_downscale2.tr 1073741824 -p lrb
    LRB	0.8954557908	0.6815446572
    ```
- for RavenH
    ```
    python3 run_cache.py ../traces/cluster29_raven_lrb_downscale10_downscale20.tr 262144 -p ravenh
    python3 run_cache.py ../traces/cluster45_raven_lrb_downscale20.tr 8388608 -p ravenh
    python3 run_cache.py ../traces/cluster17_raven_lrb_downscale10_downscale30.tr 8192 -p ravenh
    python3 run_cache.py ../traces/cluster52_raven_lrb_head3G_downscale300.tr 102400 -p ravenh
    python3 run_cache.py ../traces/cluster44_raven_lrb_downscale500.tr 10240 -p ravenh
    python3 run_cache.py ../traces/cluster24_raven_lrb_downscale300.tr 102400 -p ravenh
    python3 run_cache.py ../traces/wiki2019_remapped_downscale10_downscale20_webcachesim.tr 2147483648 -p ravenh
    python3 run_cache.py ../traces/wiki2018_downscale100_webcachesim_downscale2.tr 1073741824 -p ravenh
    ```
- for RavenL
    ```
    python3 run_cache.py ../traces/cluster29_raven_lrb_downscale10_downscale20.tr 262144 -p ravenl
    python3 run_cache.py ../traces/cluster45_raven_lrb_downscale20.tr 8388608 -p ravenl
    python3 run_cache.py ../traces/cluster17_raven_lrb_downscale10_downscale30.tr 8192 -p ravenl
    python3 run_cache.py ../traces/wiki2019_remapped_downscale10_downscale20_webcachesim.tr 2147483648 -p ravenl
    python3 run_cache.py ../traces/wiki2018_downscale100_webcachesim_downscale2.tr 1073741824 -p ravenl
    ``` 

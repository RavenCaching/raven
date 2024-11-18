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
 ```
 python3 run_cache.py $trace $cache_size -p $cache_policy
 ```


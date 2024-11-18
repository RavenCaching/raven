#!/bin/bash

#filenames=("D1a_syntheticDataset_1Sets_O1000_A0.8_Poisson_sizev1_webcachesim.tr" "D1b_syntheticDataset_1Sets_O1000_A0.8_Pareto_sizev1_webcachesim.tr" "D1d_syntheticDataset_1Sets_O1000_A0.8_Uniform_sizev1_webcachesim.tr")
#filenames=("cluster17_raven_lrb_downscale10_downscale30.tr" "cluster29_raven_lrb_downscale10_downscale20.tr"  "wiki2018_downscale100_webcachesim_downscale2.tr")  #"wiki2019_remapped_downscale10_downscale20_webcachesim.tr")
filenames=("cluster29_raven_lrb_downscale10_downscale20.tr"  "wiki2018_downscale100_webcachesim_downscale2.tr" "wiki2018_downscale100_webcachesim_downscale2.tr")
#Cs=(100 100 100)
#Cs=(8192 262144 1073741824) # 2147483648)
Cs=(131072 536870912 268435456)
objective="byte_miss_ratio"
win=1000000
batch_size=5000000
policy="ravenlensemble"  #"ravenltao" # lrb
tag="tppr_200_objall_batch5Mwind5M"  #"lookback10240"  # batch5M

cd ..
for (( i=0; i<${#filenames[*]}; ++i));
do
	python3 run_cache.py ../traces/${filenames[$i]} ${Cs[$i]} -p ${policy} -t ${tag} & 
done



 




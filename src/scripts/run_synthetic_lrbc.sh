#!/bin/bash
#filenames=("D1a_syntheticDataset_1Sets_O1000_A0.8_Poisson_sizev1_webcachesim.tr" "D1b_syntheticDataset_1Sets_O1000_A0.8_Pareto_sizev1_webcachesim.tr" "D1d_syntheticDataset_1Sets_O1000_A0.8_Uniform_sizev1_webcachesim.tr")
#filenames=("cluster17_raven_lrb_downscale10_downscale30.tr" "cluster29_raven_lrb_downscale10_downscale20.tr" "cluster52_raven_lrb_head3G_downscale300.tr" "wiki2018_downscale100_webcachesim_downscale2.tr" "wiki2019_remapped_downscale10_downscale20_webcachesim.tr")
filenames=("cluster29_raven_lrb_downscale10_downscale20.tr"  "wiki2018_downscale100_webcachesim_downscale2.tr" "wiki2018_downscale100_webcachesim_downscale2.tr")
#C=100
#Cs=(100 100 100)
#Cs=(8192 262144 102400 1073741824 2147483648)
Cs=(131072 536870912 268435456)
objective="byte_miss_ratio"
win=1000000
batch_size=5000000
n_early_stop=6000000
use_seq_to_train=1

cd ..
for (( i=0; i<${#filenames[*]}; ++i));
do
	python3 baseline_scripts.py ../traces/${filenames[$i]} ${Cs[$i]} --algorithms=lrb --memory_window=${win} --n_edc_feature=10 --no_sizeFeature=0 --no_nwithin=0 --no_LRU=0 --objective=${objective} --batch_size=${batch_size} --use_seq_to_train=${use_seq_to_train} --n_early_stop=${n_early_stop}
done




 




#!/bin/bash

runs=($(awk -F"," '{print $1}' sample_to_fit.csv))

for run in "${runs[@]}"; do
#for (( i=0; i<${#runs[@]}; i++ )); do
    #run=${runs[$i]}
    echo $run
    sbatch mendel_submit.sh ${run} /home/gvernardos/combined_posterior/input_files/ /home/gvernardos/combined_posterior/output/
done

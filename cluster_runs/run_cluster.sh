#!/bin/bash

start=$1
end=$2
path_to_cases=$3

# Activate python environment
hostname=`hostname`
if [ $hostname != "gamatos" ]
then
    module load Anaconda3
    conda init bash
    conda activate astro
else
    source $HOME/anaconda3/bin/activate astro
fi

for (( i=$start; i<=$end; i++ )); do
    printf -v name "%05d" $i
    printf "%s%s\n" "----------------------> " "$name"
    python calculate_any_system.py ${path_to_cases}/${name} > "${path_to_cases}/${name}/log.txt"
    printf "%s\n" "----------------------> DONE"
done

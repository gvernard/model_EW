#!/bin/bash

start=$1
end=$2
path_to_cases=$3

#readarray -t list < failed.dat
readarray -t list < $HOME/generate_quasar_EW_models/failed.dat


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
    name=${list[$i]}
    printf "%s%s\n" "----------------------> " "$name"
    python calculate_any_system.py ${path_to_cases}/${name} > "${path_to_cases}/${name}/log.txt"
    printf "%s\n" "----------------------> DONE"
done

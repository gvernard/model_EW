#!/bin/bash

qid=$1
inpath=$2
outpath=$3

cd /home/gvernardos/model_EW
python3.8 combined_posterior_2.py ${qid} ${inpath} ${outpath} 1> /dev/null &

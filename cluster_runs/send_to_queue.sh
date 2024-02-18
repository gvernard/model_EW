#!/bin/bash

#SBATCH --partition=shared-cpu
#SBATCH --time=7:30:00
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1000 # in MB
#SBATCH -o myjob-%A_%a.out

Nbatch=12
id_task=${SLURM_ARRAY_TASK_ID}

start=$(($id_task*$Nbatch))
stop=$(( ($id_task+1)*$Nbatch - 1 ))
srun $HOME/generate_quasar_EW_models/run_cluster_list.sh $start $stop $HOME/scratch/sdss_quasar_dps

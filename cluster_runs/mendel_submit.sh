#!/bin/bash
 
#SBATCH --partition=compute                             # Partition to use 
#SBATCH --nodes=1                                       # Number of nodes to request 
#SBATCH --tasks-per-node=1                              # Number of cores per node
#SBATCH --time=01-00:00:00                              # Format is DD-HH:MM:SS
#SBATCH --mem=512MB                                     # Memory per node. Default is 4 GB for on-prem nodes
                                                        # and aws-t2 queues. Default is 2 GB for aws-c5.4x and
                                                        # aws.c5-12x queues. Values are specified in MBs.
                                                        # Possible to also use Can use K,G,T. Setting to 0 will
                                                        # request all the memory on a node
#SBATCH --output=%x-%J.out                              # Name of file to send standard output to. You should
                                                        # also send output from your programs using their
                                                        # output options if available
#SBATCH --error=%x-%J.err                               # Name of file to send standard error to. You should
                                                        # also send errors from your programs using their
                                                        # error output options if available
QID=$1
INPATH=$2
OUTPATH=$3

module load Python/python-3.8.5

bash combined_posterior_2.sh ${QID} ${INPATH} ${OUTPATH}

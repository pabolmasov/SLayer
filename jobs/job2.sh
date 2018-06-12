#!/bin/bash
#SBATCH -A SNIC2018-5-16
#SBATCH -J slayer
#SBATCH --output=%J.out
#SBATCH --error=%J.err
#SBATCH -t 00-10:00:00
#SBATCH -n 1
#SBATCH -c 12

# export module libraries
export PYTHONPATH=$PYTHONPATH:/pfs/nobackup/home/n/natj/SLayer

# activate threading
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# go to working directory
cd /pfs/nobackup/home/n/natj/SLayer/

mpirun -np 1 python swarm.py

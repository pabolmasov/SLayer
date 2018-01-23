#!/bin/bash -l
#SBATCH -J swarm
#SBATCH -o /home/jatnat/out/%J.out
#SBATCH -e /home/jatnat/out/%J.err
#SBATCH -t 0-00:01:00 
#SBATCH -p all
#SBATCH -N 1
#SBATCH -n 12
#SBATCH --ntasks-per-node=12
#SBATCH --mem 20000
#SBATCH --mail-type=END
#SBATCH --mail-user=nattila.joonas@gmail.com
cd /home/jatnat/swarm/
export OMP_NUM_THREADS=12
python swarm.py


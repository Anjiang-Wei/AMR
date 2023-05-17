#!/bin/bash
#SBATCH -J test
#SBATCH -t 00:02:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 40
#SBATCH -p gpu
#SBATCH -o slurm.out

date
module load cuda/12.1 mpi/openmpi/4.1.5
mpirun --bind-to none ./main

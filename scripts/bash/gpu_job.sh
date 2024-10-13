#!/bin/bash
#SBATCH --account=m1727
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 1:00:00
#SBATCH -n 1
#SBATCH --gpus=1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-gpu=40000m
#SBATCH --ntasks-per-node=1

export SLURM_CPU_BIND="cores"
source activate /pscratch/sd/i/imendoza/miniconda3/envs/bpd_gpu2
srun ./vect_toy_shear_gpu.py --n-samples-gals 1000 --n-samples-shear 3000 --n-vec 100 --start-seed 1001 --end-seed 2000 --tag "gpu4_vec" --k 100 --trim 1

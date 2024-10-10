#!/bin/bash
#SBATCH --account=m1727
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -t 00:30:00
#SBATCH --ntasks-per-node=4

#ref: https://docs.nersc.gov/systems/perlmutter/running-jobs/#single-gpu-tasks-in-parallel

srun --exact -u -n 1 --gpus-per-task 1 -c 1 --mem-per-cpu=40G ./vect_toy_shear_gpu.py --n-samples-gals 1000 --n-samples-shear 3000 --n-vec 100 --start-seed 1001 --end-seed 1002 --tag "gpu2_vec" --k 100 --trim 1 &
srun --exact -u -n 1 --gpus-per-task 1 -c 1 --mem-per-cpu=40G ./vect_toy_shear_gpu.py --n-samples-gals 1000 --n-samples-shear 3000 --n-vec 100 --start-seed 1003 --end-seed 1004 --tag "gpu2_vec" --k 100 --trim 1 &
srun --exact -u -n 1 --gpus-per-task 1 -c 1 --mem-per-cpu=40G ./vect_toy_shear_gpu.py --n-samples-gals 1000 --n-samples-shear 3000 --n-vec 100 --start-seed 1005 --end-seed 1006 --tag "gpu2_vec" --k 100 --trim 1 &
srun --exact -u -n 1 --gpus-per-task 1 -c 1 --mem-per-cpu=40G ./vect_toy_shear_gpu.py --n-samples-gals 1000 --n-samples-shear 3000 --n-vec 100 --start-seed 1007 --end-seed 1008 --tag "gpu2_vec" --k 100 --trim 1 &

wait

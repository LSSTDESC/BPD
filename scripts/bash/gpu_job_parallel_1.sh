#!/bin/bash
#SBATCH --account=m1727
#SBATCH -C gpu
#SBATCH -N 1
#SBATCH -t 00:20:00
#SBATCH --ntasks-per-node=4
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=imendoza@umich.edu

#ref: https://docs.nersc.gov/systems/perlmutter/running-jobs/#single-gpu-tasks-in-parallel
K=1000
TRIM=10
TAG="gpu1_vec_n2000"

srun --exact -u -n 1 --gpus-per-task 1 -c 1 --mem-per-cpu=5G ./vect_toy_shear_gpu.py --n-samples-gals 2000 --n-samples-shear 3000 --n-vec 100 --start-seed 1 --end-seed 250 --tag $TAG --k $K --trim $TRIM &
srun --exact -u -n 1 --gpus-per-task 1 -c 1 --mem-per-cpu=5G ./vect_toy_shear_gpu.py --n-samples-gals 2000 --n-samples-shear 3000 --n-vec 100 --start-seed 251 --end-seed 500 --tag $TAG --k $K --trim $TRIM &
srun --exact -u -n 1 --gpus-per-task 1 -c 1 --mem-per-cpu=5G ./vect_toy_shear_gpu.py --n-samples-gals 2000 --n-samples-shear 3000 --n-vec 100 --start-seed 501 --end-seed 750 --tag $TAG --k $K --trim $TRIM &
srun --exact -u -n 1 --gpus-per-task 1 -c 1 --mem-per-cpu=5G ./vect_toy_shear_gpu.py --n-samples-gals 2000 --n-samples-shear 3000 --n-vec 100 --start-seed 751 --end-seed 1000 --tag $TAG --k $K --trim $TRIM &

wait

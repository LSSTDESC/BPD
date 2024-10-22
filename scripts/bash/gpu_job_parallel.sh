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
N=1_000
BASE_SEED=61
TAG="gpu1_n10000_test"

for i in $(seq 1 4);
do
    SEED="${BASE_SEED}${i}"
    CMD="python /global/u2/i/imendoza/BPD/scripts/vect_toy_shear_gpu.py --n-samples-gals ${N} --n-samples-shear 3000 --n-vec 50 --seed ${SEED} --n-seeds 250 --tag ${TAG} --k ${K} --trim ${TRIM} --sigma-e-int 2e-3"
    srun --exact -u -n 1 --gpus-per-task 1 -c 1 --mem-per-gpu=20G $CMD  &
done

wait

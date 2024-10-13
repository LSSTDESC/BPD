#!/bin/bash
#SBATCH --account=m1727
#SBATCH --nodes=1
#SBATCH --time=00:20:00
#SBATCH --constraint=cpu

conda init
conda activate bpd_cpu2
srun ./to_run.sh

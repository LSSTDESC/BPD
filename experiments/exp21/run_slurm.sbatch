#!/bin/bash
#SBATCH --job-name=exp21_42
#SBATCH --account=m1727
#SBATCH --qos=debug
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=imendoza@umich.edu
#SBATCH -C gpu
#SBATCH --output=/global/u2/i/imendoza/BPD/jobs_out/exp21_42_%j.out
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

./run_inference_galaxy_images.py 42

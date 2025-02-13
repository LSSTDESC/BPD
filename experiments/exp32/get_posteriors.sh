#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"
export JAX_ENABLE_X64="True"
SEED="43"
 
./get_interim_samples.py $SEED
../../scripts/get_shear_all_free.py $SEED --overwrite

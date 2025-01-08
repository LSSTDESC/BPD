#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"
export JAX_ENABLE_X64="True"
SEED="43"
 
./get_image_interim_samples_fixed.py $SEED
../../scripts/get_shear_from_interim_samples.py $SEED exp30_$SEED "e_post_${SEED}.npz" --overwrite

#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"
export JAX_ENABLE_X64="True"
SEED="43"
 
./get_interim_samples.py $SEED
../../scripts/get_shear_from_shapes.py $SEED --old-seed $SEED --interim-samples-fname "e_post_${SEED}.npz" --tag exp31_ --overwrite

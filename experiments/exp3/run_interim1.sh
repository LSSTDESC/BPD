#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"


./get_image_interim_samples_fixed1.py 43 interim_image_fix3  --n-gals 500 --n-vec 100
./get_shear_from_interim_samples.py 43 interim_image_fix3 "e_post_43.npz"

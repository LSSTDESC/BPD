#!/bin/bash
export CUDA_VISIBLE_DEVICES="0"
export JAX_ENABLE_X64="True"

 
./get_image_interim_samples_fixed.py 43
../../scripts/get_shear_from_interim_samples.py 43 test_fixed_shear_inference_images_43 "e_post_42.npz"

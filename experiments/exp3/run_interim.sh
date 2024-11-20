#!/bin/bash
export CUDA_VISIBLE_DEVICES="1"
export JAX_ENABLE_X64="True"



./get_image_interim_samples_fixed.py 42
# ../../scripts/get_shear_from_inter_samples.py 42 test_fixed_shear_inference_images_42 "e_post_42.npz"

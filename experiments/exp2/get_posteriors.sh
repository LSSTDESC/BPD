#!/bin/bash
export CUDA_VISIBLE_DEVICES="1"
export JAX_ENABLE_X64="True"

./run_inference_galaxy_images.py 42

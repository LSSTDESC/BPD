#!/bin/bash
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS="cpu"
export JAX_ENABLE_X64="True"

./make_figures.py

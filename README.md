# BPD: Bayesian Pixel Domain Shear Inference

Bayesian Pixel Domain shear estimation based on automatically differentiable cell-based coadd modeling. 

This repository contains functions to run HMC (Hamiltonian Monte Carlo) using [JAX-Galsim](https://github.com/GalSim-developers/JAX-GalSim) as a forward model to perform shear inference. 

## Installation

```bash
# fresh conda env
pip install --upgrade pip
conda create -n bpd python=3.12 # only 3.10, 3.11, and 3.12 supported.
conda activate bpd

# Install JAX (on GPU)
git clone git@github.com:LSSTDESC/BPD.git
cd BPD
pip install -U "jax[cuda12]<0.7.0"
pip install -e .
pip install -e ".[dev]"

# Install JAX-Galsim
pip install git+https://github.com/GalSim-developers/JAX-GalSim.git --no-deps --no-build-isolation

# Might be necessary
conda install -c nvidia cuda-nvcc
```

# BPD: Bayesian Pixel Domain Shear Inference

Bayesian Pixel Domain shear estimation based on automatically differentiable cell-based coadd modeling. 

This repository contains functions to run HMC (Hamiltonian Monte Carlo) using [JAX-Galsim](https://github.com/GalSim-developers/JAX-GalSim) as a forward model to perform shear inference. 


## Installation

```bash
# fresh conda env
pip install --upgrade pip
conda create -n bpd python=3.12
conda activate bpd

# Install JAX (cuda)
pip install -U "jax[cuda12]"

# Install JAX-Galsim
pip install git+https://github.com/GalSim-developers/JAX-GalSim.git

# Install package and depedencies
git clone git@github.com:LSSTDESC/BPD.git
cd BPD
python -m pip install . -e
python -m pip install .[dev]
```

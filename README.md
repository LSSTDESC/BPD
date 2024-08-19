# BPD: Bayesian Pixel Domain Shear Inference

Bayesian Pixel Domain shear estimation based on automatically differentiable cell-based coadd modeling. 

This repository contains functions to run HMC (Hamiltonian Monte Carlo) using [JAX-Galsim](https://github.com/GalSim-developers/JAX-GalSim) as a forward model to perform galaxy parameter inference. 


## Installation

```bash
# fresh conda env
pip install --upgrade pip
conda create -n bpd python=3.10
conda activate bpd

# Install JAX
pip install -U "jax[cuda12]"

# descwl-shear-sims dependencies
conda install -c conda-forge mamba
mamba install -c conda-forge stackvana
mamba install -c conda-forge pip lsstdesc.weaklensingdeblending numba galsim ipykernel ngmix

pip install git+https://github.com/LSSTDESC/descwl-shear-sims.git
pip install git+https://github.com/esheldon/metadetect.git
pip install git+https://github.com/GalSim-developers/JAX-GalSim.git

pip install numpyro
pip install blackjax
pip install ChainConsumer arviz

cd BPD
pip install -e .
```

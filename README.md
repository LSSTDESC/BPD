# BPD: Bayesian Pixel Domain Shear Inference

Bayesian Pixel Domain shear estimation based on automatically differentiable cell-based coadd modeling


## Installation

```bash
# fresh conda env
pip install --upgrade pip
conda create -n bpd python=3.10
conda activate bpd

# Install JAX
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# descwl-shear-sims dependencies
conda install -c conda-forge mamba
mamba install -c conda-forge stackvana
mamba install -c conda-forge pip lsstdesc.weaklensingdeblending numba galsim ipykernel ngmix

pip install git+https://github.com/LSSTDESC/descwl-shear-sims.git
pip install git+https://github.com/esheldon/metadetect.git
pip install git+https://github.com/GalSim-developers/JAX-GalSim.git

pip install numpyro
pip install h5py

cd BPD
pip install -e .
```

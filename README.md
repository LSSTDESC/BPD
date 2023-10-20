# BPD: Bayesian Pixel Domain Shear Inference

Bayesian Pixel Domain shear estimation based on automatically differentiable cell-based coadd modeling


## Installation

```bash
pip install --upgrade pip
conda create -n bpd python=3.10
conda activate bpd
conda install -c conda-forge mamba
mamba install -c conda-forge stackvana
mamba install -c conda-forge pip lsstdesc.weaklensingdeblending numba galsim ipykernel ngmix

# pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# NERSC specific
# module load cudatoolkit/12.0
# module load cudnn/8.9.3_cuda12
# pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

pip install git+https://github.com/LSSTDESC/descwl-shear-sims.git
pip install git+https://github.com/esheldon/metadetect.git
pip install git+https://github.com/GalSim-developers/JAX-GalSim.git

pip install numpyro

# BPD 
cd BPD
pip install -e .
```

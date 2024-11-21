# Experiment 2

This folder contains scripts to reproduce timing and convergence results on fitting galaxy images
with NUTS with a realistic noise settings and broad priors on parameters. We initialize chains from
true prior rather than from a ball or fixing them to true parameters. 
We don't perform shear inference.

* SNR ~ (8, 100)
* shape noise = 0.3 (realistic)
* Shear = (0.02, 0.0)
* Fix warmup steps = 500
* Number of samples = 1000

## Results

- Contour plots for parameters of galaxies 
- Trace plots for paramteres of galaxies
- Convergence histograms (rhat, ESS)
- Timing as a function of chains 

## How to reproduce figures

```bash
./get_posteriors.sh
./make_figures.py
```

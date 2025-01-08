# Experiment 3.0

This folder contains scripts to reproduce results on fitting galaxy images in the low noise setting
and inferring shear from them. Details: 

* Only the shapes vary during inference, other parameters are fixed to truth.
* The flux is fixed to `1e6` and hlr to `1.0`, so that the galaxies achieve ~ 1000 SNR with background of `1.0`.
* We dither the galaxies within the center pixel and impose a interim and true prior of `N(0, 0.5)`.
* We use shape noise of `1e-4`.
* We use a total of `1000` galaxies.


## How to reproduce results

```bash
./run_interim.sh
```

## Figures



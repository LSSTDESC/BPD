# Experiment 4.0

Shear inference on images with more or less realistic noise conditions, only ellipticities are free.

- SNR is centered around ~15
- 10^4 galaxies
- Use jackknife + shape noise cancellation to estimate mean and error on mean shear.
- HLR is fixed to `0.8` where images of size `63` are sufficient.

## Reproducing results

```bash 
make samples # wait for slurm job to finish
make collate 
make shear 
make figures
```

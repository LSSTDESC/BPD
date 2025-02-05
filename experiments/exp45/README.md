# Experiment 4.5

Same as experiment `4.4`, but SNR is centered around 100. Also first multi-gpu run experiment.

## Reproducing results

```bash 
make samples # wait for slurm job to finish
make collate 
make shear 
make figures
```

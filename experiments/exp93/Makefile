SHELL=/bin/bash
SEED := 51
EXP := exp93
TAG := ${EXP}_${SEED}
DIR := /pscratch/sd/i/imendoza/data/cache_chains/${TAG}
SHEAR := 0.02


# image sampling phase 
QOS := regular
TIME_SAMPLE := 02:00
NODES_SAMPLE := 8
MEM_PER_GPU := 10G
N_GALS := 10_000
N_SAMPLES_PER_GAL := 300

export JAX_ENABLE_X64=True



samples_plus_nu01: 
	./slurm_get_interim_samples.py ${SEED}1 --tag ${TAG} --g1 ${SHEAR} --g2 0.0 --n-gals ${N_GALS} --n-samples-per-gal ${N_SAMPLES_PER_GAL} --time ${TIME_SAMPLE} --mode "plus" --mem-per-gpu ${MEM_PER_GPU} --qos ${QOS} --nodes ${NODES_SAMPLE} --nu 0.1

samples_minus_nu01:
	./slurm_get_interim_samples.py ${SEED}1 --tag ${TAG} --g1 -${SHEAR} --g2 0.0 --n-gals ${N_GALS} --n-samples-per-gal ${N_SAMPLES_PER_GAL} --time ${TIME_SAMPLE} --mode "minus" --mem-per-gpu ${MEM_PER_GPU} --qos ${QOS} --nodes ${NODES_SAMPLE} --nu 0.1


test_slurm_samples:
	./slurm_get_interim_samples.py 991 --tag ${TAG} --g1 ${SHEAR} --g2 0.0 --n-gals 1000 --n-samples-per-gal 300 --mode "plus" --batch-size 1000 --qos "debug" --nodes 1 --time 00:10 --nu 0.1

test_samples_simple:
	./get_interim_samples.py 990 --tag ${TAG} --g1 ${SHEAR} --g2 0.0 --n-gals 100 --n-samples-per-gal 300 --mode "plus" --batch-size 100 --nu 0.1

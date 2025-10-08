SHELL=/bin/bash
SEED := 51
EXP := exp91
TAG := ${EXP}_${SEED}
DIR := /pscratch/sd/i/imendoza/data/cache_chains/${TAG}

# simulation parameters
N_GALS := 10_000
N_SAMPLES_PER_GAL := 300
SHEAR := 0.02
QOS := regular

TIME_SAMPLE := 02:00
NODES_SAMPLE := 8
MEM_PER_GPU := 10G

PLUS_SAMPLES_FPATH := ${DIR}/interim_samples_${SEED}1_plus.npz
MINUS_SAMPLES_FPATH := ${DIR}/interim_samples_${SEED}1_minus.npz

export JAX_ENABLE_X64=True

error:
	../exp72/simple_error.py ${SEED}3 --tag ${TAG} --samples-plus-fpath ${PLUS_SAMPLES_FPATH} --samples-minus-fpath ${MINUS_SAMPLES_FPATH}

shear_plus: 
	export CUDA_VISIBLE_DEVICES=0
	../exp72/get_shear.py ${SEED}2 --samples-fpath ${PLUS_SAMPLES_FPATH} --tag ${TAG} --overwrite --mode "plus"

shear_minus:
	export CUDA_VISIBLE_DEVICES=0
	../exp72/get_shear.py ${SEED}2 --samples-fpath ${MINUS_SAMPLES_FPATH} --tag ${TAG} --overwrite --mode "minus"

collate_plus:
	../exp70/collate_samples.py --tag ${TAG} --mode "plus" --start-string "interim_samples_${SEED}1"

collate_minus:
	../exp70/collate_samples.py --tag ${TAG} --mode "minus" --start-string "interim_samples_${SEED}1"

samples_plus: 
	./slurm_get_interim_samples.py ${SEED}1 --tag ${TAG} --g1 ${SHEAR} --g2 0.0 --n-gals ${N_GALS} --n-samples-per-gal ${N_SAMPLES_PER_GAL} --time ${TIME_SAMPLE} --mode "plus" --mem-per-gpu ${MEM_PER_GPU} --qos ${QOS} --nodes ${NODES_SAMPLE}

samples_minus:
	./slurm_get_interim_samples.py ${SEED}1 --tag ${TAG} --g1 -${SHEAR} --g2 0.0 --n-gals ${N_GALS} --n-samples-per-gal ${N_SAMPLES_PER_GAL} --time ${TIME_SAMPLE} --mode "minus" --mem-per-gpu ${MEM_PER_GPU} --qos ${QOS} --nodes ${NODES_SAMPLE}

test_slurm_samples:
	./slurm_get_interim_samples.py 992 --tag ${TAG} --g1 ${SHEAR} --g2 0.0 --n-gals 1000 --n-samples-per-gal 300 --mode "plus" --batch-size 1000 --qos "debug" --nodes 1 --time 00:10

test_samples:
	./get_interim_samples.py 991 --tag ${TAG} --g1 ${SHEAR} --g2 0.0 --n-gals 1000 --n-samples-per-gal 300 --mode "plus" --batch-size 1000


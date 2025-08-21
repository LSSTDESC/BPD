SHELL=/bin/bash
SEED := 51
EXP := exp72
TAG := ${EXP}_${SEED}
DIR := /pscratch/sd/i/imendoza/data/cache_chains/${TAG}

# simulation parameters
N_GALS := 10_000
N_SAMPLES_PER_GAL := 300
SHEAR := 0.02
QOS := regular

TIME_SAMPLE := 02:00
NODES_SAMPLE := 8
NTASKS_SAMPLE := 4
MEM_PER_GPU := 10G

PLUS_SAMPLES_FPATH := ${DIR}/interim_samples_${SEED}1_plus.npz
MINUS_SAMPLES_FPATH := ${DIR}/interim_samples_${SEED}1_minus.npz

export JAX_ENABLE_X64=True

figures:
	../exp70/get_figures.py ${SEED} --tag ${TAG}

error:
	./simple_error.py ${SEED}4 --tag ${TAG} --samples-plus-fpath ${PLUS_SAMPLES_FPATH} --samples-minus-fpath ${MINUS_SAMPLES_FPATH}

verify_error:
	./simple_error.py 111 --tag ${TAG} --samples-plus-fpath ${PLUS_SAMPLES_FPATH} --samples-minus-fpath ${MINUS_SAMPLES_FPATH} --n-gals 10_000

collate_boot:
	../exp70/collate_samples.py 111 --tag ${TAG} --mode "" --start-string "g_samples_boots"

boot:
	./slurm_bootstrap.py 111 --tag ${TAG} --samples-plus-fpath ${PLUS_SAMPLES_FPATH} --samples-minus-fpath ${MINUS_SAMPLES_FPATH} --n-boots 200 --n-gals 10_000

shear_plus: 
	export CUDA_VISIBLE_DEVICES=0
	./get_shear.py ${SEED}2 --samples-fpath ${PLUS_SAMPLES_FPATH} --tag ${TAG} --overwrite --mode "plus"

shear_minus:
	export CUDA_VISIBLE_DEVICES=0
	./get_shear.py ${SEED}2 --samples-fpath ${MINUS_SAMPLES_FPATH} --tag ${TAG} --overwrite --mode "minus"

collate_plus:
	../exp70/collate_samples.py ${SEED}1 --tag ${TAG} --mode "plus"

collate_minus:
	../exp70/collate_samples.py ${SEED}1 --tag ${TAG} --mode "minus"

samples_plus: 
	./slurm_get_interim_samples.py ${SEED}1 --tag ${TAG} --g1 ${SHEAR} --g2 0.0 --n-gals ${N_GALS} --n-samples-per-gal ${N_SAMPLES_PER_GAL} --time ${TIME_SAMPLE} --mode "plus" --mem-per-gpu ${MEM_PER_GPU} --qos ${QOS} --nodes ${NODES_SAMPLE} --n-tasks-per-node ${NTASKS_SAMPLE}

samples_minus:
	./slurm_get_interim_samples.py ${SEED}1 --tag ${TAG} --g1 -${SHEAR} --g2 0.0 --n-gals ${N_GALS} --n-samples-per-gal ${N_SAMPLES_PER_GAL} --time ${TIME_SAMPLE} --mode "minus" --mem-per-gpu ${MEM_PER_GPU} --qos ${QOS} --nodes ${NODES_SAMPLE}  --n-tasks-per-node ${NTASKS_SAMPLE}

test_sample:
	./slurm_get_interim_samples.py 111 --tag ${TAG} --g1 0.02 --g2 0.0 --n-gals 10_000 --n-samples-per-gal 300 --time 02:00 --mode "plus" --mem-per-gpu 10G --qos regular --nodes 1 --n-tasks-per-node 1

clean: 
	rm -f ${DIR}/g_samples_*.npy ${DIR}/interim_samples_*.npz ${DIR}/g_samples_*.npz

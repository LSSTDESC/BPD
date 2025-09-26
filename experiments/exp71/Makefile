SHELL=/bin/bash
SEED := 51
EXP := exp71
TAG := ${EXP}_${SEED}
DIR := /pscratch/sd/i/imendoza/data/cache_chains/${TAG}

# simulation parameters
SHEAR := 0.02
QOS := debug
MEM_PER_GPU := 10G

PLUS_FILE := /pscratch/sd/i/imendoza/data/cache_chains/exp70_51/interim_samples_511_plus.npz
MINUS_FILE := /pscratch/sd/i/imendoza/data/cache_chains/exp70_51/interim_samples_511_minus.npz

export JAX_ENABLE_X64=True

boot2:
	export CUDA_VISIBLE_DEVICES=0
	./slurm_bootstrap.py ${SEED}52 --tag ${TAG} --samples-plus-fpath ${PLUS_FILE}  --samples-minus-fpath ${MINUS_FILE} --n-boots 300 --n-nodes 3 --qos regular --time 08:00

boot1:
	export CUDA_VISIBLE_DEVICES=0
	./slurm_bootstrap.py ${SEED}51 --tag ${TAG} --samples-plus-fpath ${PLUS_FILE}  --samples-minus-fpath ${MINUS_FILE} --n-boots 200 --n-nodes 2 --qos regular --time 06:00

collate_boot1:
	../exp70/collate_samples.py --tag ${TAG} --mode "" --start-string "g_samples_boots_${SEED}51"

# main pipeline
error: 
	./simple_error.py ${SEED}4 --tag ${TAG} --plus-samples-fpath ${PLUS_FILE} --minus-samples-fpath ${MINUS_FILE}

collate_boot:
	../exp70/collate_samples.py ${SEED}3 --tag ${TAG} --mode "" --start-string "g_samples_boots"

shear_plus: 
	export CUDA_VISIBLE_DEVICES=0
	../../scripts/get_shear_and_sn_from_shapes.py ${SEED}2 --samples-fpath ${PLUS_FILE} --tag ${TAG} --overwrite --extra-tag "plus"
	
shear_minus:
	export CUDA_VISIBLE_DEVICES=0
	../../scripts/get_shear_and_sn_from_shapes.py ${SEED}2 --samples-fpath ${MINUS_FILE} --tag ${TAG} --overwrite --extra-tag "minus"

# Testing
test_boot2:
	export CUDA_VISIBLE_DEVICES=0
	./slurm_bootstrap.py 992 --tag ${TAG} --samples-plus-fpath ${PLUS_FILE} --samples-minus-fpath ${MINUS_FILE} --n-boots 8 --n-nodes 1 --time 00:30 --qos debug --mem-per-gpu 10G

test_boot1:
	export CUDA_VISIBLE_DEVICES=0
	./simple_bootstrap.py 991 --tag ${TAG} --samples-plus-fpath ${PLUS_FILE} --samples-minus-fpath ${MINUS_FILE} --n-boots 2 --n-gals 100

# Misc.
clean: 
	rm -f ${DIR}/g_samples_*.npy ${DIR}/interim_samples_*.npz ${DIR}/g_samples_*.npz

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

error: 
	./simple_error.py ${SEED}4 --tag ${TAG} --plus-samples-fpath ${PLUS_FILE} --minus-samples-fpath ${MINUS_FILE}

collate_boot:
	../exp70/collate_samples.py ${SEED}3 --tag ${TAG} --mode "" --start-string "g_samples_boots"

bootstrap:
	export CUDA_VISIBLE_DEVICES=0
	./slurm_bootstrap.py ${SEED}3 --tag ${TAG} --samples-plus-fpath ${PLUS_FILE}  --samples-minus-fpath ${MINUS_FILE} --n-boots 200

shear_plus: 
	export CUDA_VISIBLE_DEVICES=0
	../../scripts/get_shear_and_sn_from_shapes.py ${SEED}2 --samples-fpath ${PLUS_FILE} --tag ${TAG} --overwrite --extra-tag "plus"
	
shear_minus:
	export CUDA_VISIBLE_DEVICES=0
	../../scripts/get_shear_and_sn_from_shapes.py ${SEED}2 --samples-fpath ${MINUS_FILE} --tag ${TAG} --overwrite --extra-tag "minus"


clean: 
	rm -f ${DIR}/g_samples_*.npy ${DIR}/interim_samples_*.npz ${DIR}/g_samples_*.npz

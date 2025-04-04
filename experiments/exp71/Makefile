SHELL=/bin/bash
SEED := 44
EXP := exp70
TAG := ${EXP}_${SEED}
DIR := /pscratch/sd/i/imendoza/data/cache_chains/${TAG}

# simulation parameters
SHEAR := 0.02
QOS := debug
MEM_PER_GPU := 10G

PLUS_FILE := /pscratch/sd/i/imendoza/data/cache_chains/exp70_44/interim_samples_441_plus.npz
MINUS_FILE := /pscratch/sd/i/imendoza/data/cache_chains/exp70_44/interim_samples_441_minus.npz

export JAX_ENABLE_X64=True

figures:
	../exp70/get_figures.py ${SEED} --tag ${TAG}

collate_boot:
	../exp70/collate_samples.py ${SEED} --tag ${TAG} --mode "" --start-string "g_samples_boots"

bootstrap:
	export CUDA_VISIBLE_DEVICES=0
	./slurm_bootstrap.py ${SEED} --tag ${TAG} --samples-plus-fpath ${PLUS_FILE}  --samples-minus-fpath ${MINUS_FILE}

shear: 
	export CUDA_VISIBLE_DEVICES=0
	../../scripts/get_shear_and_sn_from_shapes.py ${SEED}2 --interim-samples-fname "interim_samples_${SEED}_plus.npz" --tag ${TAG} --overwrite --extra-tag "plus"
	
	../../scripts/get_shear_and_sn_from_shapes.py ${SEED}2 --interim-samples-fname "interim_samples_${SEED}_minus.npz" --tag ${TAG} --overwrite --extra-tag "minus"


clean: 
	rm -f ${DIR}/g_samples_*.npy ${DIR}/interim_samples_*.npz ${DIR}/g_samples_*.npz

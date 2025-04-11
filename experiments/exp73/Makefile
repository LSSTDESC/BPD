SHELL=/bin/bash
SEED := 44
EXP := exp73
TAG := ${EXP}_${SEED}
DIR := /pscratch/sd/i/imendoza/data/cache_chains/${TAG}

PLUS_FILE := /pscratch/sd/i/imendoza/data/cache_chains/exp72_44/interim_samples_451_plus.npz
MINUS_FILE := /pscratch/sd/i/imendoza/data/cache_chains/exp72_44/interim_samples_451_minus.npz

export JAX_ENABLE_X64=True

figures:
	../exp70/get_figures.py ${SEED} --tag ${TAG}

error:
	./simple_error.py ${SEED}4 --tag ${TAG} --samples-plus-fpath ${PLUS_FILE} --samples-minus-fpath ${MINUS_FILE}

shear: 
	export CUDA_VISIBLE_DEVICES=0
	../../scripts/get_shear_all_free.py ${SEED}2 --interim-samples-fpath ${DIR}/"interim_samples_${SEED}1_plus.npz" --tag ${TAG} --overwrite --mode "plus"
	../../scripts/get_shear_all_free.py ${SEED}2 --interim-samples-fpath ${DIR}/"interim_samples_${SEED}1_minus.npz" --tag ${TAG} --overwrite --mode "minus"


clean: 
	rm -f ${DIR}/g_samples_*.npy ${DIR}/interim_samples_*.npz ${DIR}/g_samples_*.npz

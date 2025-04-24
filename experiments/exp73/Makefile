SHELL=/bin/bash
SEED := 51
EXP := exp73
TAG := ${EXP}_${SEED}
DIR := /pscratch/sd/i/imendoza/data/cache_chains/${TAG}

PLUS_FILE := /pscratch/sd/i/imendoza/data/cache_chains/exp72_51/interim_samples_511_plus.npz
MINUS_FILE := /pscratch/sd/i/imendoza/data/cache_chains/exp72_51/interim_samples_511_minus.npz

export JAX_ENABLE_X64=True

figures:
	../exp70/get_figures.py ${SEED} --tag ${TAG}

slurm_error_post:
	./slurm_simple_error2.py ${SEED}4 --tag ${TAG} --samples-plus-fpath ${PLUS_FILE} --samples-minus-fpath ${MINUS_FILE} --posterior-fpath ${DIR}/shear_samples_512_plus.npz

error_post:
	./simple_error2.py ${SEED}4 --tag ${TAG} --samples-plus-fpath ${PLUS_FILE} --samples-minus-fpath ${MINUS_FILE} --posterior-fpath ${DIR}/shear_samples_512_plus.npz

slurm_shear:
	./slurm_get_shear.py ${SEED}2 --samples-fpath ${PLUS_FILE} --tag ${TAG} --extra-tag "plus"
	./slurm_get_shear.py ${SEED}2 --samples-fpath ${MINUS_FILE} --tag ${TAG} --extra-tag "minus"


clean: 
	rm -f ${DIR}/g_samples_*.npy ${DIR}/interim_samples_*.npz ${DIR}/g_samples_*.npz

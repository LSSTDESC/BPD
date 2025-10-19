SHELL=/bin/bash
SEED := 51
EXP := exp92
TAG := ${EXP}_${SEED}

PLUS_SAMPLES_FPATH := /pscratch/sd/i/imendoza/data/cache_chains/exp91_51/interim_samples_511_plus.npz
MINUS_SAMPLES_FPATH := /pscratch/sd/i/imendoza/data/cache_chains/exp91_51/interim_samples_511_minus.npz

export JAX_ENABLE_X64=True

subset_error:
	../exp73/slurm_simple_error.py ${SEED}3 --tag ${TAG} --samples-plus-fpath ${PLUS_SAMPLES_FPATH} --samples-minus-fpath ${MINUS_SAMPLES_FPATH}

shear_samples:
	../exp73/slurm_get_shear.py ${SEED}2 --samples-fpath ${PLUS_SAMPLES_FPATH} --tag ${TAG} --extra-tag "plus"
	../exp73/slurm_get_shear.py ${SEED}2 --samples-fpath ${MINUS_SAMPLES_FPATH} --tag ${TAG} --extra-tag "minus"

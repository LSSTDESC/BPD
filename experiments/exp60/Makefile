SHELL=/bin/bash
SEED := 42
EXP := exp60
TAG := ${EXP}_${SEED}
DIR := /pscratch/sd/i/imendoza/data/cache_chains/${TAG}

export JAX_ENABLE_X64=True

figures: 
	./get_figures.py ${SEED} --tag ${TAG}

shear: 
	export CUDA_VISIBLE_DEVICES=0
	./get_shear_and_sn_from_shapes.py ${SEED} --samples-fname /pscratch/sd/i/imendoza/data/cache_chains/exp45_${SEED}/interim_samples_${SEED}_plus.npz --tag ${TAG} --overwrite --extra-tag "plus"
	./get_shear_and_sn_from_shapes.py ${SEED} --samples-fname /pscratch/sd/i/imendoza/data/cache_chains/exp45_${SEED}/interim_samples_${SEED}_minus.npz --tag ${TAG} --overwrite --extra-tag "minus"

clean: 
	rm -f ${DIR}/g_samples_*.npy ${DIR}/g_samples_*.npz

SHELL=/bin/bash
SEED := 43
DIR := /pscratch/sd/i/imendoza/data/cache_chains/exp40_${SEED}
TAG := exp40_${SEED}
export JAX_ENABLE_X64="True"

figures:
	./get_figures.py ${SEED}

shear: 
	export CUDA_VISIBLE_DEVICES="0"
	../../scripts/get_shear_from_shapes.py ${SEED} --old-seed ${SEED} --interim-samples-fname "interim_samples_${SEED}_plus.npz" --tag ${TAG} --overwrite --extra-tag "plus"
	../../scripts/get_shear_from_shapes.py ${SEED} --old-seed ${SEED} --interim-samples-fname "interim_samples_${SEED}_minus.npz" --tag ${TAG} --overwrite --extra-tag "minus"

collate:
	./collate_samples.py ${SEED} --tag ${TAG} --mode "plus" --n-files 4
	./collate_samples.py ${SEED} --tag ${TAG} --mode "minus" --n-files 4

samples: 
	./slurm_get_interim_samples.py ${SEED} --tag ${TAG}  --g1 0.02 --g2 0.0 --mode "plus"
	./slurm_get_interim_samples.py ${SEED} --tag ${TAG} --g1 -0.02 --g2 0.0 --mode "minus"

clean: 
	rm -f ${DIR}/g_samples_*.npy ${DIR}/interim_samples_*.npz

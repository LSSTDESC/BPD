SHELL=/bin/bash
SEED := 42
DIR := /pscratch/sd/i/imendoza/data/cache_chains/exp40_${SEED}
export JAX_ENABLE_X64="True"

figures:
	./make_figures.py --seed ${SEED}

shear: 
	export CUDA_VISIBLE_DEVICES="0"
	../../scripts/get_shear_from_shapes.py ${SEED} --old-seed ${SEED} --interim-samples-fname "interim_samples_${SEED}_plus.npz" --tag exp40_${SEED} --overwrite --extra-tag "plus"
	../../scripts/get_shear_from_shapes.py ${SEED} --old-seed ${SEED} --interim-samples-fname "interim_samples_${SEED}_minus.npz" --tag exp40_${SEED} --overwrite --extra-tag "minus"

collate:
	./collate_samples.py --seed ${SEED} --mode "plus" --n-files 4
	./collate_samples.py --seed ${SEED} --mode "minus" --n-files 4

samples: 
	./slurm_get_interim_samples.py 42 --g1 0.02 --g2 0.0 --mode "plus"
	./slurm_get_interim_samples.py 42 --g1 -0.02 --g2 0.0 --mode "minus"

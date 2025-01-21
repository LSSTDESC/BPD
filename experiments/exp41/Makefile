SHELL=/bin/bash
SEED := 42
DIR := /pscratch/sd/i/imendoza/data/cache_chains/exp41_${SEED}
TAG := exp41_${SEED}
export JAX_ENABLE_X64=True

figures:
	./get_figures.py ${SEED} --tag ${TAG}

jackknife:
	export CUDA_VISIBLE_DEVICES=0
	./get_shear_jackknife.py ${SEED} --old-seed ${SEED} \
	--samples-plus-fname interim_samples_${SEED}_plus.npz \
	--samples-minus-fname interim_samples_${SEED}_minus.npz \
	--tag ${TAG} --overwrite

shear: 
	export CUDA_VISIBLE_DEVICES=0
	../../scripts/get_shear_from_shapes.py ${SEED} --old-seed ${SEED} --interim-samples-fname "interim_samples_${SEED}_plus.npz" --tag ${TAG} --overwrite --extra-tag "plus"
	../../scripts/get_shear_from_shapes.py ${SEED} --old-seed ${SEED} --interim-samples-fname "interim_samples_${SEED}_minus.npz" --tag ${TAG} --overwrite --extra-tag "minus"

collate:
	./collate_samples.py ${SEED} --tag ${TAG} --mode "plus" --n-files 4
	./collate_samples.py ${SEED} --tag ${TAG} --mode "minus" --n-files 4

samples: 
	../exp40/slurm_get_interim_samples.py ${SEED} --tag ${TAG}  --g1 0.02 --g2 0.0 --n-gals 250 --n-samples-per-gal 150 --mean-logflux 6.0 --sigma-logflux 0.1 --shape-noise 1e-3 --sigma-e-int 0.04 --time "00:15" --mode "plus"
	../exp40/slurm_get_interim_samples.py ${SEED} --tag ${TAG}  --g1 -0.02 --g2 0.0 --n-gals 250 --n-samples-per-gal 150 --mean-logflux 6.0 --sigma-logflux 0.1 --shape-noise 1e-3 --sigma-e-int 0.04 --time "00:15 --mode "minus"

clean: 
	rm -f ${DIR}/g_samples_*.npy ${DIR}/interim_samples_*.npz

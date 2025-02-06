SHELL=/bin/bash
SEED := 42
EXP := exp44
TAG := ${EXP}_${SEED}
DIR := /pscratch/sd/i/imendoza/data/cache_chains/${TAG}

# simulation parameters
N_GALS := 2500
N_SAMPLES_PER_GAL := 300
SHAPE_NOISE := 0.01
SIGMA_E_INT := 0.05
MEAN_LOG_FLUX := 4.5
SIGMA_LOG_FLUX := 0.3
SHEAR := 0.02
QOS := debug

export JAX_ENABLE_X64=True

figures:
	../exp40/get_figures.py ${SEED} --tag ${TAG}

jackknife:
	export CUDA_VISIBLE_DEVICES=0
	../exp40/get_shear_jackknife.py ${SEED} --old-seed ${SEED} \
	--samples-plus-fname interim_samples_${SEED}_plus.npz \
	--samples-minus-fname interim_samples_${SEED}_minus.npz \
	--tag ${TAG} --overwrite

shear: 
	export CUDA_VISIBLE_DEVICES=0
	../../scripts/get_shear_from_shapes.py ${SEED} --old-seed ${SEED} --interim-samples-fname "interim_samples_${SEED}_plus.npz" --tag ${TAG} --overwrite --extra-tag "plus"
	../../scripts/get_shear_from_shapes.py ${SEED} --old-seed ${SEED} --interim-samples-fname "interim_samples_${SEED}_minus.npz" --tag ${TAG} --overwrite --extra-tag "minus"

collate:
	../exp40/collate_samples.py ${SEED} --tag ${TAG} --mode "plus"
	../exp40/collate_samples.py ${SEED} --tag ${TAG} --mode "minus"

samples: 
	../exp40/slurm_get_interim_samples.py ${SEED} --tag ${TAG} --g1 ${SHEAR} --g2 0.0 --n-gals ${N_GALS} --n-samples-per-gal ${N_SAMPLES_PER_GAL} --mean-logflux ${MEAN_LOG_FLUX} --sigma-logflux ${SIGMA_LOG_FLUX} --shape-noise ${SHAPE_NOISE} --sigma-e-int ${SIGMA_E_INT} --time "00:25" --mode "plus" --mem-per-gpu "10G" --qos ${QOS}
	../exp40/slurm_get_interim_samples.py ${SEED} --tag ${TAG} --g1 -${SHEAR} --g2 0.0 --n-gals ${N_GALS} --n-samples-per-gal ${N_SAMPLES_PER_GAL} --mean-logflux ${MEAN_LOG_FLUX} --sigma-logflux ${SIGMA_LOG_FLUX} --shape-noise ${SHAPE_NOISE} --sigma-e-int ${SIGMA_E_INT} --time "00:25" --mode "minus" --mem-per-gpu "10G" --qos ${QOS}

clean: 
	rm -f ${DIR}/g_samples_*.npy ${DIR}/interim_samples_*.npz ${DIR}/g_samples_*.npz

SHELL=/bin/bash
SEED := 42
EXP := exp50
TAG := ${EXP}_${SEED}
DIR := /pscratch/sd/i/imendoza/data/cache_chains/${TAG}

# simulation parameters
N_GALS := 2500
N_SAMPLES_PER_GAL := 300
SHAPE_NOISE := 0.01
SIGMA_E_INT := 0.05
MEAN_LOG_FLUX := 4.0
SIGMA_LOG_FLUX := 0.3
MEAN_LOG_HLR := -0.1
SIGMA_LOG_HLR := 0.05
SHEAR := 0.02
QOS := debug

export JAX_ENABLE_X64=True

figures:
	../exp50/get_figures.py ${SEED} --tag ${TAG}

jackknife:
	export CUDA_VISIBLE_DEVICES=0
	../exp50/get_shear_jackknife.py ${SEED} --old-seed ${SEED} \
	--samples-plus-fname interim_samples_${SEED}_plus.npz \
	--samples-minus-fname interim_samples_${SEED}_minus.npz \
	--tag ${TAG} --overwrite

shear: 
	export CUDA_VISIBLE_DEVICES=0
	../../scripts/get_shear_all_free.py ${SEED} --samples-fname "interim_samples_${SEED}_plus.npz" --tag ${TAG} --overwrite --mode "plus"
	../../scripts/get_shear_all_free.py ${SEED} --samples-fname "interim_samples_${SEED}_minus.npz" --tag ${TAG} --overwrite --mode "minus"

collate:
	../exp50/collate_samples.py ${SEED} --tag ${TAG} --mode "plus"
	../exp50/collate_samples.py ${SEED} --tag ${TAG} --mode "minus"

samples: 
	../exp50/slurm_get_interim_samples.py ${SEED} --tag ${TAG} --g1 ${SHEAR} --g2 0.0 --n-gals ${N_GALS} --n-samples-per-gal ${N_SAMPLES_PER_GAL} --mean-logflux ${MEAN_LOG_FLUX} --sigma-logflux ${SIGMA_LOG_FLUX} --mean-loghlr ${MEAN_LOG_HLR} --sigma-loghlr ${SIGMA_LOG_HLR} --shape-noise ${SHAPE_NOISE} --sigma-e-int ${SIGMA_E_INT} --time "00:25" --mode "plus" --mem-per-gpu "10G" --qos ${QOS}
	../exp50/slurm_get_interim_samples.py ${SEED} --tag ${TAG} --g1 -${SHEAR} --g2 0.0 --n-gals ${N_GALS} --n-samples-per-gal ${N_SAMPLES_PER_GAL} --mean-logflux ${MEAN_LOG_FLUX} --sigma-logflux ${SIGMA_LOG_FLUX} --mean-loghlr ${MEAN_LOG_HLR} --sigma-loghlr ${SIGMA_LOG_HLR} --shape-noise ${SHAPE_NOISE} --sigma-e-int ${SIGMA_E_INT} --time "00:25" --mode "minus" --mem-per-gpu "10G" --qos ${QOS}

clean: 
	rm -f ${DIR}/g_samples_*.npy ${DIR}/interim_samples_*.npz ${DIR}/g_samples_*.npz

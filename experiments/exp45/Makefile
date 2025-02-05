SHELL=/bin/bash
SEED := 42
EXP := exp45
TAG := ${EXP}_${SEED}
DIR := /pscratch/sd/i/imendoza/data/cache_chains/${TAG}

# simulation parameters
N_GALS := 625
N_SAMPLES_PER_GAL := 300
SHAPE_NOISE := 0.01
SIGMA_E_INT := 0.05
MEAN_LOG_FLUX := 4.0
SIGMA_LOG_FLUX := 0.3
SHEAR := 0.02
QOS := debug
NODES_SAMPLE := 4
NTASKS_SAMPLE := 4


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
	../exp40/collate_samples.py ${SEED} --tag ${TAG} --mode "plus" --n-files 4
	../exp40/collate_samples.py ${SEED} --tag ${TAG} --mode "minus" --n-files 4

samples: 
	../exp40/slurm_get_interim_samples.py ${SEED} --tag ${TAG} --g1 ${SHEAR} --g2 0.0 --n-gals ${N_GALS} --n-samples-per-gal ${N_SAMPLES_PER_GAL} --mean-logflux ${MEAN_LOG_FLUX} --sigma-logflux ${SIGMA_LOG_FLUX} --shape-noise ${SHAPE_NOISE} --sigma-e-int ${SIGMA_E_INT} --time "00:10" --mode "plus" --mem-per-gpu "5G" --qos ${QOS} --nodes ${NODES_SAMPLE} --n-tasks-per-node ${NTASKS_SAMPLE}
	../exp40/slurm_get_interim_samples.py ${SEED} --tag ${TAG} --g1 -${SHEAR} --g2 0.0 --n-gals ${N_GALS} --n-samples-per-gal ${N_SAMPLES_PER_GAL} --mean-logflux ${MEAN_LOG_FLUX} --sigma-logflux ${SIGMA_LOG_FLUX} --shape-noise ${SHAPE_NOISE} --sigma-e-int ${SIGMA_E_INT} --time "00:10" --mode "minus" --mem-per-gpu "5G" --qos ${QOS} --nodes ${NODES_SAMPLE}  --n-tasks-per-node ${NTASKS_SAMPLE}


clean: 
	rm -f ${DIR}/g_samples_*.npy ${DIR}/interim_samples_*.npz ${DIR}/g_samples_*.npz

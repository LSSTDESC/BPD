SHELL=/bin/bash
SEED := 42
EXP := exp70
TAG := ${EXP}_${SEED}
DIR := /pscratch/sd/i/imendoza/data/cache_chains/${TAG}

# simulation parameters
N_GALS := 625
N_SAMPLES_PER_GAL := 300
SHEAR := 0.02
QOS := debug

TIME_SAMPLE := 00:10
NODES_SAMPLE := 4
NTASKS_SAMPLE := 4
MEM_PER_GPU := 10G


export JAX_ENABLE_X64=True

figures:
	../exp70/get_figures.py ${SEED} --tag ${TAG}

collate_jack:
	../exp70/collate_samples.py ${SEED} --tag ${TAG} --mode "" --start-string "g_samples_jack"

jackknife:
	export CUDA_VISIBLE_DEVICES=0
	./slurm_jackknife.py ${SEED} --tag ${TAG}

shear: 
	export CUDA_VISIBLE_DEVICES=0
	../../scripts/get_shear_and_sn_from_shapes.py ${SEED} --interim-samples-fname "interim_samples_${SEED}_plus.npz" --tag ${TAG} --overwrite --extra-tag "plus"
	
	../../scripts/get_shear_and_sn_from_shapes.py ${SEED} --interim-samples-fname "interim_samples_${SEED}_minus.npz" --tag ${TAG} --overwrite --extra-tag "minus"


clean: 
	rm -f ${DIR}/g_samples_*.npy ${DIR}/interim_samples_*.npz ${DIR}/g_samples_*.npz

SHELL=/bin/bash
SEED := 50
EXP := exp70
TAG := ${EXP}_${SEED}
DIR := /pscratch/sd/i/imendoza/data/cache_chains/${TAG}

# simulation parameters
N_GALS := 5000
N_SAMPLES_PER_GAL := 300
SHEAR := 0.02
QOS := debug

TIME_SAMPLE := 00:30
NODES_SAMPLE := 15
NTASKS_SAMPLE := 4
MEM_PER_GPU := 10G


export JAX_ENABLE_X64=True

figures:
	../exp70/get_figures.py ${SEED} --tag ${TAG}

collate_boot:
	../exp70/collate_samples.py ${SEED}3 --tag ${TAG} --mode "" --start-string "g_samples_boots"

boot:
	./slurm_bootstrap.py ${SEED}3 --tag ${TAG} --samples-plus-fname "interim_samples_${SEED}1_plus.npz" --samples-minus-fname "interim_samples_${SEED}1_minus.npz"

shear: 
	export CUDA_VISIBLE_DEVICES=0
	time ../../scripts/get_shear_from_shapes.py ${SEED}2 --interim-samples-fname "interim_samples_${SEED}1_plus.npz" --tag ${TAG} --overwrite --extra-tag "plus"
	
	../../scripts/get_shear_from_shapes.py ${SEED}2 --interim-samples-fname "interim_samples_${SEED}1_minus.npz" --tag ${TAG} --overwrite --extra-tag "minus"

collate:
	../exp70/collate_samples.py ${SEED}1 --tag ${TAG} --mode "plus"
	../exp70/collate_samples.py ${SEED}1 --tag ${TAG} --mode "minus"

samples: 
	./slurm_get_interim_samples.py ${SEED}1 --tag ${TAG} --g1 ${SHEAR} --g2 0.0 --n-gals ${N_GALS} --n-samples-per-gal ${N_SAMPLES_PER_GAL} --time ${TIME_SAMPLE} --mode "plus" --mem-per-gpu ${MEM_PER_GPU} --qos ${QOS} --nodes ${NODES_SAMPLE} --n-tasks-per-node ${NTASKS_SAMPLE}

	./slurm_get_interim_samples.py ${SEED}1 --tag ${TAG} --g1 -${SHEAR} --g2 0.0 --n-gals ${N_GALS} --n-samples-per-gal ${N_SAMPLES_PER_GAL} --time ${TIME_SAMPLE} --mode "minus" --mem-per-gpu ${MEM_PER_GPU} --qos ${QOS} --nodes ${NODES_SAMPLE}  --n-tasks-per-node ${NTASKS_SAMPLE}


clean: 
	rm -f ${DIR}/g_samples_*.npy ${DIR}/interim_samples_*.npz ${DIR}/g_samples_*.npz

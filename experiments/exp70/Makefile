SHELL=/bin/bash
SEED := 51
EXP := exp70
TAG := ${EXP}_${SEED}
DIR := /pscratch/sd/i/imendoza/data/cache_chains/${TAG}

# simulation parameters
N_GALS := 10_000
N_SAMPLES_PER_GAL := 300
SHEAR := 0.02
QOS := regular

TIME_SAMPLE := 02:00
NODES_SAMPLE := 8
NTASKS_SAMPLE := 4
MEM_PER_GPU := 10G


export JAX_ENABLE_X64=True

figures:
	../exp70/get_figures.py ${SEED} --tag ${TAG}

collate_boot:
	../exp70/collate_samples.py 111 --tag ${TAG} --mode "" --start-string "g_samples_boots"

test_boot2:
	./slurm_bootstrap.py 992 --tag ${TAG} --samples-plus-fname "interim_samples_${SEED}1_plus.npz" --samples-minus-fname "interim_samples_${SEED}1_minus.npz" --time 00:30 --n-boots 8 --n-nodes 1 --qos debug

test_boot:
	export CUDA_VISIBLE_DEVICES=0
	./simple_bootstrap.py 991 --tag ${TAG} --samples-plus-fname "interim_samples_${SEED}1_plus.npz" --samples-minus-fname "interim_samples_${SEED}1_minus.npz"

new_boot2:
	./slurm_bootstrap.py ${SEED}52 --tag ${TAG} --samples-plus-fname "interim_samples_${SEED}1_plus.npz" --samples-minus-fname "interim_samples_${SEED}1_minus.npz" --time 04:00 --n-boots 300 --n-nodes 3 --qos regular

new_boot2:
	./slurm_bootstrap.py ${SEED}51 --tag ${TAG} --samples-plus-fname "interim_samples_${SEED}1_plus.npz" --samples-minus-fname "interim_samples_${SEED}1_minus.npz" --time 04:00 --n-boots 300 --n-nodes 2 --qos regular

boot:
	./slurm_bootstrap.py 111 --tag ${TAG} --samples-plus-fname "interim_samples_${SEED}1_plus.npz" --samples-minus-fname "interim_samples_${SEED}1_minus.npz" --n-gals 10_000 --time 00:30

verify_error:
	./simple_error.py 111 --tag ${TAG} --plus-samples-fname "interim_samples_${SEED}1_plus.npz" --minus-samples-fname "interim_samples_${SEED}1_minus.npz" --n-gals 10_000

error:
	./simple_error.py ${SEED}4 --tag ${TAG} --plus-samples-fname "interim_samples_${SEED}1_plus.npz" --minus-samples-fname "interim_samples_${SEED}1_minus.npz"

shear_plus: 
	export CUDA_VISIBLE_DEVICES=0
	../../scripts/get_shear_from_shapes.py ${SEED}2 --interim-samples-fname "interim_samples_${SEED}1_plus.npz" --tag ${TAG} --overwrite --extra-tag "plus"

shear_minus:
	export CUDA_VISIBLE_DEVICES=0
	../../scripts/get_shear_from_shapes.py ${SEED}2 --interim-samples-fname "interim_samples_${SEED}1_minus.npz" --tag ${TAG} --overwrite --extra-tag "minus"

collate_plus:
	../exp70/collate_samples.py ${SEED}1 --tag ${TAG} --mode "plus"

collate_minus:
	../exp70/collate_samples.py ${SEED}1 --tag ${TAG} --mode "minus"

samples_plus:
	./slurm_get_interim_samples.py ${SEED}1 --tag ${TAG} --g1 ${SHEAR} --g2 0.0 --n-gals ${N_GALS} --n-samples-per-gal ${N_SAMPLES_PER_GAL} --time ${TIME_SAMPLE} --mode "plus" --mem-per-gpu ${MEM_PER_GPU} --qos ${QOS} --nodes ${NODES_SAMPLE} --n-tasks-per-node ${NTASKS_SAMPLE}

samples_minus:
	./slurm_get_interim_samples.py ${SEED}1 --tag ${TAG} --g1 -${SHEAR} --g2 0.0 --n-gals ${N_GALS} --n-samples-per-gal ${N_SAMPLES_PER_GAL} --time ${TIME_SAMPLE} --mode "minus" --mem-per-gpu ${MEM_PER_GPU} --qos ${QOS} --nodes ${NODES_SAMPLE}  --n-tasks-per-node ${NTASKS_SAMPLE}

test_samples: 
	./slurm_get_interim_samples.py ${SEED}1 --tag ${TAG} --g1 0.02 --g2 0.0 --n-gals 10_000 --n-samples-per-gal 100 --time 02:00 --mode "plus" --mem-per-gpu ${MEM_PER_GPU} --qos "regular" --nodes 1 --n-tasks-per-node 1

clean: 
	rm -f ${DIR}/g_samples_*.npy ${DIR}/interim_samples_*.npz ${DIR}/g_samples_*.npz

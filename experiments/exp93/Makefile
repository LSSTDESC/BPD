SHELL=/bin/bash
SEED := 51
EXP := exp93
TAG := ${EXP}_${SEED}
DIR := /pscratch/sd/i/imendoza/data/cache_chains/${TAG}
SHEAR := 0.02


# image sampling phase 
QOS := regular
TIME_SAMPLE := 02:00
NODES_SAMPLE := 8
MEM_PER_GPU := 10G
N_GALS := 10_000
N_SAMPLES_PER_GAL := 300

export JAX_ENABLE_X64=True

PLUS_SAMPLES_FPATH_NU01 = ${DIR}/interim_samples_${SEED}1_plus_nu10.npz
MINUS_SAMPLES_FPATH_NU01 = ${DIR}/interim_samples_${SEED}1_minus_nu10.npz

PLUS_SAMPLES_FPATH_NU03 = ${DIR}/interim_samples_${SEED}1_plus_nu30.npz
MINUS_SAMPLES_FPATH_NU03 = ${DIR}/interim_samples_${SEED}1_minus_nu30.npz

shear_nu01: 
	export CUDA_VISIBLE_DEVICES=0
	../exp72/get_shear.py ${SEED}2 --samples-fpath ${PLUS_SAMPLES_FPATH_NU01} --tag ${TAG} --overwrite --mode "plus"
	../exp72/get_shear.py ${SEED}2 --samples-fpath ${MINUS_SAMPLES_FPATH_NU01} --tag ${TAG} --overwrite --mode "minus"

shear_nu03: 
	export CUDA_VISIBLE_DEVICES=0
	../exp72/get_shear.py ${SEED}2 --samples-fpath ${PLUS_SAMPLES_FPATH_NU03} --tag ${TAG} --overwrite --mode "plus"
	../exp72/get_shear.py ${SEED}2 --samples-fpath ${MINUS_SAMPLES_FPATH_NU03} --tag ${TAG} --overwrite --mode "minus"

collate_interim_nu01:
	../exp70/collate_samples.py --tag ${TAG} --mode "plus" --start-string "interim_samples_${SEED}1" --contains "nu10"
	../exp70/collate_samples.py --tag ${TAG} --mode "minus" --start-string "interim_samples_${SEED}1" --contains "nu10"

collate_interim_nu03:
	../exp70/collate_samples.py --tag ${TAG} --mode "plus" --start-string "interim_samples_${SEED}1" --contains "nu30"
	../exp70/collate_samples.py --tag ${TAG} --mode "minus" --start-string "interim_samples_${SEED}1" --contains "nu30"

samples_nu03: 
	./slurm_get_interim_samples.py ${SEED}1 --tag ${TAG} --g1 ${SHEAR} --g2 0.0 --n-gals ${N_GALS} --n-samples-per-gal ${N_SAMPLES_PER_GAL} --time ${TIME_SAMPLE} --mode "plus" --mem-per-gpu ${MEM_PER_GPU} --qos ${QOS} --nodes ${NODES_SAMPLE} --nu 0.3

	./slurm_get_interim_samples.py ${SEED}1 --tag ${TAG} --g1 -${SHEAR} --g2 0.0 --n-gals ${N_GALS} --n-samples-per-gal ${N_SAMPLES_PER_GAL} --time ${TIME_SAMPLE} --mode "minus" --mem-per-gpu ${MEM_PER_GPU} --qos ${QOS} --nodes ${NODES_SAMPLE} --nu 0.3


samples_plus_nu01: 
	./slurm_get_interim_samples.py ${SEED}1 --tag ${TAG} --g1 ${SHEAR} --g2 0.0 --n-gals ${N_GALS} --n-samples-per-gal ${N_SAMPLES_PER_GAL} --time ${TIME_SAMPLE} --mode "plus" --mem-per-gpu ${MEM_PER_GPU} --qos ${QOS} --nodes ${NODES_SAMPLE} --nu 0.1

samples_minus_nu01:
	./slurm_get_interim_samples.py ${SEED}1 --tag ${TAG} --g1 -${SHEAR} --g2 0.0 --n-gals ${N_GALS} --n-samples-per-gal ${N_SAMPLES_PER_GAL} --time ${TIME_SAMPLE} --mode "minus" --mem-per-gpu ${MEM_PER_GPU} --qos ${QOS} --nodes ${NODES_SAMPLE} --nu 0.1


test_slurm_samples:
	./slurm_get_interim_samples.py 991 --tag ${TAG} --g1 ${SHEAR} --g2 0.0 --n-gals 1000 --n-samples-per-gal 300 --mode "plus" --batch-size 1000 --qos "debug" --nodes 1 --time 00:10 --nu 0.1

test_samples_simple:
	./get_interim_samples.py 990 --tag ${TAG} --g1 ${SHEAR} --g2 0.0 --n-gals 100 --n-samples-per-gal 300 --mode "plus" --batch-size 100 --nu 0.1

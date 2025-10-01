SHELL=/bin/bash
SEED := 51
EXP := exp73
TAG := ${EXP}_${SEED}
DIR := /pscratch/sd/i/imendoza/data/cache_chains/${TAG}

PLUS_FILE := /pscratch/sd/i/imendoza/data/cache_chains/exp72_51/interim_samples_511_plus.npz
MINUS_FILE := /pscratch/sd/i/imendoza/data/cache_chains/exp72_51/interim_samples_511_minus.npz

export JAX_ENABLE_X64=True

boot_slurm1:
	./slurm_bootstrap.py ${SEED}51 --tag ${TAG} --samples-plus-fpath ${PLUS_FILE}  --samples-minus-fpath ${MINUS_FILE} --n-boots 100 --n-nodes 5 --qos regular --time 20:00

test_boot_slurm2:
	./slurm_bootstrap.py 994 --tag ${TAG} --samples-plus-fpath ${PLUS_FILE}  --samples-minus-fpath ${MINUS_FILE} --n-boots 8 --n-nodes 1 --qos regular --time 06:00

test_boot_slurm1:
	./slurm_bootstrap.py 993 --tag ${TAG} --samples-plus-fpath ${PLUS_FILE}  --samples-minus-fpath ${MINUS_FILE} --n-boots 8 --n-nodes 1 --qos debug --time 00:30 --n-samples 10 --n-warmup-steps 10

test_boot2: # memory
	./simple_bootstrap.py 992 --tag ${TAG} --samples-plus-fpath ${PLUS_FILE}  --samples-minus-fpath ${MINUS_FILE} --n-boots 2 --n-gals 300000 --n-samples 10 --n-warmup-steps 10

test_boot1:
	./simple_bootstrap.py 991 --tag ${TAG} --samples-plus-fpath ${PLUS_FILE}  --samples-minus-fpath ${MINUS_FILE} --n-boots 2 --n-gals 100 --n-samples 10

figures:
	../exp70/get_figures.py ${SEED} --tag ${TAG}

collate_error_post:
	../exp70/collate_samples.py ${SEED}4 --tag ${TAG} --mode "" --start-string "g_samples_514_errs"

slurm_error_post:
	./slurm_simple_error2.py ${SEED}4 --tag ${TAG} --samples-plus-fpath ${PLUS_FILE} --samples-minus-fpath ${MINUS_FILE} --posterior-fpath ${DIR}/shear_samples_512_plus.npz

error_post:
	./simple_error2.py ${SEED}4 --tag ${TAG} --samples-plus-fpath ${PLUS_FILE} --samples-minus-fpath ${MINUS_FILE} --posterior-fpath ${DIR}/shear_samples_512_plus.npz

slurm_shear:
	./slurm_get_shear.py ${SEED}2 --samples-fpath ${PLUS_FILE} --tag ${TAG} --extra-tag "plus"
	./slurm_get_shear.py ${SEED}2 --samples-fpath ${MINUS_FILE} --tag ${TAG} --extra-tag "minus"


clean: 
	rm -f ${DIR}/g_samples_*.npy ${DIR}/interim_samples_*.npz ${DIR}/g_samples_*.npz

SHELL=/bin/bash
SEED := 42
EXP := exp22
TAG := ${EXP}_${SEED}
DIR := /pscratch/sd/i/imendoza/data/cache_chains/${TAG}

export JAX_ENABLE_X64=True

figures:
	./get_figures.py ${SEED} ${TAG}

convergence:
	./get_convergence.py ${SEED} ${TAG}

collate: 
	./collate_samples.py ${SEED} --tag ${TAG}

samples:
	./slurm_full_samples.py ${SEED} ${TAG}

clean: 
	rm -f ${DIR}/*.npz ${DIR}/*.npy

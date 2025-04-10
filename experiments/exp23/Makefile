SHELL=/bin/bash
SEED := 43
EXP := exp23
TAG := ${EXP}_${SEED}
DIR := /pscratch/sd/i/imendoza/data/cache_chains/${TAG}

export JAX_ENABLE_X64=True

figures:
	./get_figures.py ${SEED} ${TAG}

convergence:
	./get_convergence.py ${SEED} ${TAG}

collate: 
	./collate_samples.py ${SEED} --tag ${TAG}

timing:
	./slurm_run_timing.py ${SEED} ${TAG}

full:
	./slurm_full_samples.py ${SEED} ${TAG}

clean: 
	rm -f ${DIR}/*.npz ${DIR}/*.npy

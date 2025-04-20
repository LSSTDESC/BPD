SHELL=/bin/bash
SEED := 64
EXP := exp80
TAG := ${EXP}_${SEED}
DIR := /pscratch/sd/i/imendoza/data/cache_chains/${TAG}

SHAPE_NOISE = 0.2
SIGMA_E_INT = 0.3
N_GALS = 30_000

export JAX_ENABLE_X64=True

collate_mc: 
	../exp70/collate_samples.py ${SEED}4 --tag ${TAG} --start-string mc_exp_samples --mode ""

collate: 
	../exp70/collate_samples.py ${SEED}2 --tag ${TAG} --start-string g_samples_boots --mode ""

monte_carlo:
	./slurm_mc_shear.py ${SEED}4 --tag ${TAG} --shape-noise ${SHAPE_NOISE} --sigma-e-int ${SIGMA_E_INT} --n-gals ${N_GALS}

jack: 
	./slurm_jackknife.py ${SEED}3 --tag ${TAG}

boot: 
	./slurm_bootstrap.py ${SEED}2 --tag ${TAG} --time "00:29"

samples: 
	export CUDA_VISIBLE_DEVICES=0
	./get_shape_samples.py ${SEED}1 ${TAG} --shape-noise ${SHAPE_NOISE} --sigma-e-int ${SIGMA_E_INT} --n-gals ${N_GALS}

clean: 
	rm -f ${DIR}/*

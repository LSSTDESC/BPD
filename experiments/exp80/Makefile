SHELL=/bin/bash
SEED := 42
EXP := exp80
TAG := ${EXP}_${SEED}
DIR := /pscratch/sd/i/imendoza/data/cache_chains/${TAG}

export JAX_ENABLE_X64=True

collate: 
	../exp70/collate_samples.py ${SEED}2 --tag ${TAG} --start-string g_samples_boots --mode ""

boot: 
	./slurm_bootstrap.py ${SEED}2 --tag ${TAG}

samples: 
	export CUDA_VISIBLE_DEVICES=0
	./get_shape_samples.py ${SEED}1 ${TAG}


clean: 
	rm -f ${DIR}/*

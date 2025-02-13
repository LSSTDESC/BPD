SHELL=/bin/bash
SEED := 42
EXP := exp21
TAG := ${EXP}_${SEED}
DIR := /pscratch/sd/i/imendoza/data/cache_chains/${TAG}

export JAX_ENABLE_X64=True

figures:
	./get_figures.py

convergence:
	./get_convergence.py ${SEED} ${TAG}

samples:
	export CUDA_VISIBLE_DEVICES=0
	./run_inference_galaxy_images.py ${SEED} ${TAG}

clean: 
	rm -f ${DIR}/chain_results*.npy

SHELL=/bin/bash
SEED := 42
DIR := /pscratch/sd/i/imendoza/data/cache_chains/exp40_${SEED}

vpath %.npz ${DIR}
vpath %.npy ${DIR}
vpath %.pdf figs

contours_bias.pdf: figs g_samples_${SEED}_${SEED}_plus.npy g_samples_${SEED}_${SEED}_minus.npy
	touch figs/contours_bias.pdf

figs: 
	mkdir figs

g_samples_${SEED}_${SEED}_plus.npy:
	test -f ${DIR}/interim_samples_${SEED}_plus.npz
	../../scripts/get_shear_from_shapes.py ${SEED} --old-seed ${SEED} --interim-samples-fname "interim_samples_${SEED}_plus.npz" --tag exp40_${SEED} --overwrite --extra-tag "plus"

g_samples_${SEED}_${SEED}_minus.npy: interim_samples_${SEED}_minus.npz
	test -f ${DIR}/interim_samples_${SEED}_plus.npz
	../../scripts/get_shear_from_shapes.py ${SEED} --old-seed ${SEED} --interim-samples-fname "interim_samples_${SEED}_minus.npz" --tag exp40_${SEED} --overwrite --extra-tag "minus"


# contours_bias.pdf: figs g_samples_${SEED}_${SEED}_plus.npy g_samples_${SEED}_${SEED}_minus.npy
# 	./make_figures.py --seed ${SEED}

# figs: 
# 	mkdir figs

# g_samples_${SEED}_${SEED}_plus.npy: interim_samples_${SEED}_plus.npz
# 	test -f ${DIR}/interim_samples_${SEED}_plus.npz
# 	../../scripts/get_shear_from_shapes.py ${SEED} --old-seed ${SEED} --interim-samples-fname "interim_samples_${SEED}_plus.npz" --tag exp40_${SEED} --overwrite --extra-tag "plus"

# g_samples_${SEED}_${SEED}_minus.npy: interim_samples_${SEED}_minus.npz
# 	test -f ${DIR}/interim_samples_${SEED}_plus.npz
# 	../../scripts/get_shear_from_shapes.py ${SEED} --old-seed ${SEED} --interim-samples-fname "interim_samples_${SEED}_minus.npz" --tag exp40_${SEED} --overwrite --extra-tag "minus"

# interim_samples_${SEED}_plus.npz: interim_samples_${SEED}%_plus.npz
# 	test -f ${DIR}/interim_samples_${SEED}0_plus.npz
# 	./collate_samples.py --seed ${SEED} --mode "plus" --n-files 4

# interim_samples_${SEED}_minus.npz: interim_samples_${SEED}%_minus.npz 
# 	test -f ${DIR}/interim_samples_${SEED}0_minus.npz
# 	./collate_samples.py --seed ${SEED} --mode "minus" --n-files 4

# interim_samples_${SEED}%_plus.npz: ${DIR}
# 	./slurm_get_interim_samples.py 42 --g1 0.02 --g2 0.0 --mode "plus"

# interim_samples_${SEED}%_minus.npz: ${DIR}
# 	./slurm_get_interim_samples.py 42 --g1 -0.02 --g2 0.0 --mode "minus"

${DIR}:
	mkdir ${DIR}


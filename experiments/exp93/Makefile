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

PLUS_SAMPLES_FPATH_NU05 = ${DIR}/interim_samples_${SEED}1_plus_nu50.npz
MINUS_SAMPLES_FPATH_NU05 = ${DIR}/interim_samples_${SEED}1_minus_nu50.npz

PLUS_SAMPLES_FPATH_NU10 = ${DIR}/interim_samples_${SEED}1_plus_nu100.npz
MINUS_SAMPLES_FPATH_NU10 = ${DIR}/interim_samples_${SEED}1_minus_nu100.npz

PLUS_SAMPLES_FPATH_NU20 = ${DIR}/interim_samples_${SEED}1_plus_nu200.npz
MINUS_SAMPLES_FPATH_NU20 = ${DIR}/interim_samples_${SEED}1_minus_nu200.npz

PLUS_SAMPLES_FPATH_NU30 = ${DIR}/interim_samples_${SEED}1_plus_nu300.npz
MINUS_SAMPLES_FPATH_NU30 = ${DIR}/interim_samples_${SEED}1_minus_nu300.npz

PLUS_SAMPLES_FPATH_NU-02 = ${DIR}/interim_samples_${SEED}1_plus_nu-20.npz
MINUS_SAMPLES_FPATH_NU-02 = ${DIR}/interim_samples_${SEED}1_minus_nu-20.npz

PLUS_SAMPLES_FPATH_NU-04 = ${DIR}/interim_samples_${SEED}1_plus_nu-40.npz
MINUS_SAMPLES_FPATH_NU-04 = ${DIR}/interim_samples_${SEED}1_minus_nu-40.npz

PLUS_SAMPLES_FPATH_NU-06 = ${DIR}/interim_samples_${SEED}1_plus_nu-60.npz
MINUS_SAMPLES_FPATH_NU-06 = ${DIR}/interim_samples_${SEED}1_minus_nu-60.npz


shear_samples: 
	export CUDA_VISIBLE_DEVICES=0
# 	../exp72/get_shear.py ${SEED}2 --samples-fpath ${PLUS_SAMPLES_FPATH_NU01} --tag ${TAG} --mode "plus" --extra "nu10"
# 	../exp72/get_shear.py ${SEED}2 --samples-fpath ${MINUS_SAMPLES_FPATH_NU01} --tag ${TAG} --mode "minus" --extra "nu10"

# 	../exp72/get_shear.py ${SEED}2 --samples-fpath ${PLUS_SAMPLES_FPATH_NU03} --tag ${TAG} --mode "plus" --extra "nu30"
# 	../exp72/get_shear.py ${SEED}2 --samples-fpath ${MINUS_SAMPLES_FPATH_NU03} --tag ${TAG} --mode "minus" --extra "nu30"

# 	../exp72/get_shear.py ${SEED}2 --samples-fpath ${PLUS_SAMPLES_FPATH_NU05} --tag ${TAG} --mode "plus" --extra "nu50"
# 	../exp72/get_shear.py ${SEED}2 --samples-fpath ${MINUS_SAMPLES_FPATH_NU05} --tag ${TAG} --mode "minus" --extra "nu50"


	../exp72/get_shear.py ${SEED}2 --samples-fpath ${PLUS_SAMPLES_FPATH_NU10} --tag ${TAG} --mode "plus" --extra "nu100"
	../exp72/get_shear.py ${SEED}2 --samples-fpath ${MINUS_SAMPLES_FPATH_NU10} --tag ${TAG} --mode "minus" --extra "nu100"


	../exp72/get_shear.py ${SEED}2 --samples-fpath ${PLUS_SAMPLES_FPATH_NU20} --tag ${TAG} --mode "plus" --extra "nu200"
	../exp72/get_shear.py ${SEED}2 --samples-fpath ${MINUS_SAMPLES_FPATH_NU20} --tag ${TAG} --mode "minus" --extra "nu200"

	../exp72/get_shear.py ${SEED}2 --samples-fpath ${PLUS_SAMPLES_FPATH_NU30} --tag ${TAG} --mode "plus" --extra "nu300"
	../exp72/get_shear.py ${SEED}2 --samples-fpath ${MINUS_SAMPLES_FPATH_NU30} --tag ${TAG} --mode "minus" --extra "nu300"


	../exp72/get_shear.py ${SEED}2 --samples-fpath ${PLUS_SAMPLES_FPATH_NU-02} --tag ${TAG} --mode "plus" --extra "nu-20"
	../exp72/get_shear.py ${SEED}2 --samples-fpath ${MINUS_SAMPLES_FPATH_NU-02} --tag ${TAG} --mode "minus" --extra "nu-20"


	../exp72/get_shear.py ${SEED}2 --samples-fpath ${PLUS_SAMPLES_FPATH_NU-04} --tag ${TAG} --mode "plus" --extra "nu-40"
	../exp72/get_shear.py ${SEED}2 --samples-fpath ${MINUS_SAMPLES_FPATH_NU-04} --tag ${TAG} --mode "minus" --extra "nu-40"

	../exp72/get_shear.py ${SEED}2 --samples-fpath ${PLUS_SAMPLES_FPATH_NU-06} --tag ${TAG} --mode "plus" --extra "nu-60"
	../exp72/get_shear.py ${SEED}2 --samples-fpath ${MINUS_SAMPLES_FPATH_NU-06} --tag ${TAG} --mode "minus" --extra "nu-60"

boot:
# 	../exp72/slurm_bootstrap.py ${SEED}3 --tag ${TAG} --samples-plus-fpath ${PLUS_SAMPLES_FPATH_NU05} --samples-minus-fpath ${MINUS_SAMPLES_FPATH_NU05} --n-boots 100 --extra "nu50" --time "03:00" --qos "regular" --mem-per-gpu 40G
	../exp72/slurm_bootstrap.py ${SEED}3 --tag ${TAG} --samples-plus-fpath ${PLUS_SAMPLES_FPATH_NU01} --samples-minus-fpath ${MINUS_SAMPLES_FPATH_NU01} --n-boots 100 --extra "nu10" --time "03:00" --qos "regular" --mem-per-gpu 40G

	../exp72/slurm_bootstrap.py ${SEED}3 --tag ${TAG} --samples-plus-fpath ${PLUS_SAMPLES_FPATH_NU03} --samples-minus-fpath ${MINUS_SAMPLES_FPATH_NU03} --n-boots 100 --extra "nu30" --time "03:00" --qos "regular" --mem-per-gpu 40G

	../exp72/slurm_bootstrap.py ${SEED}3 --tag ${TAG} --samples-plus-fpath ${PLUS_SAMPLES_FPATH_NU10} --samples-minus-fpath ${MINUS_SAMPLES_FPATH_NU10} --n-boots 100 --extra "nu100" --time "03:00" --qos "regular" --mem-per-gpu 40G
	
	../exp72/slurm_bootstrap.py ${SEED}3 --tag ${TAG} --samples-plus-fpath ${PLUS_SAMPLES_FPATH_NU20} --samples-minus-fpath ${MINUS_SAMPLES_FPATH_NU20} --n-boots 100 --extra "nu200" --time "03:00" --qos "regular" --mem-per-gpu 40G

	../exp72/slurm_bootstrap.py ${SEED}3 --tag ${TAG} --samples-plus-fpath ${PLUS_SAMPLES_FPATH_NU30} --samples-minus-fpath ${MINUS_SAMPLES_FPATH_NU30} --n-boots 100 --extra "nu300" --time "03:00" --qos "regular" --mem-per-gpu 40G

	../exp72/slurm_bootstrap.py ${SEED}3 --tag ${TAG} --samples-plus-fpath ${PLUS_SAMPLES_FPATH_NU-02} --samples-minus-fpath ${MINUS_SAMPLES_FPATH_NU-02} --n-boots 100 --extra "nu-20" --time "03:00" --qos "regular" --mem-per-gpu 40G

	../exp72/slurm_bootstrap.py ${SEED}3 --tag ${TAG} --samples-plus-fpath ${PLUS_SAMPLES_FPATH_NU-04} --samples-minus-fpath ${MINUS_SAMPLES_FPATH_NU-04} --n-boots 100 --extra "nu-40" --time "03:00" --qos "regular" --mem-per-gpu 40G

	../exp72/slurm_bootstrap.py ${SEED}3 --tag ${TAG} --samples-plus-fpath ${PLUS_SAMPLES_FPATH_NU-06} --samples-minus-fpath ${MINUS_SAMPLES_FPATH_NU-06} --n-boots 100 --extra "nu-60" --time "03:00" --qos "regular" --mem-per-gpu 40G

collate_boot:
# 	../exp70/collate_samples.py --tag ${TAG} --start-string "g_samples_boots_${SEED}3" --contains "nu50" --mode ""
 
	../exp70/collate_samples.py --tag ${TAG} --start-string "g_samples_boots_${SEED}3" --contains "nu10" --mode ""

	../exp70/collate_samples.py --tag ${TAG} --start-string "g_samples_boots_${SEED}3" --contains "nu30" --mode ""

# 	../exp70/collate_samples.py --tag ${TAG} --start-string "g_samples_boots_${SEED}3" --contains "nu100" --mode ""

# 	../exp70/collate_samples.py --tag ${TAG} --start-string "g_samples_boots_${SEED}3" --contains "nu200" --mode ""

# 	../exp70/collate_samples.py --tag ${TAG} --start-string "g_samples_boots_${SEED}3" --contains "nu300" --mode ""

# 	../exp70/collate_samples.py --tag ${TAG} --start-string "g_samples_boots_${SEED}3" --contains "nu-20" --mode ""

# 	../exp70/collate_samples.py --tag ${TAG} --start-string "g_samples_boots_${SEED}3" --contains "nu-40" --mode ""

# 	../exp70/collate_samples.py --tag ${TAG} --start-string "g_samples_boots_${SEED}3" --contains "nu-60" --mode ""


test_slurm_boot:
	../exp72/slurm_bootstrap.py 992 --tag ${TAG} --samples-plus-fpath ${PLUS_SAMPLES_FPATH_NU05} --samples-minus-fpath ${MINUS_SAMPLES_FPATH_NU05} --n-boots 8 --extra "nu50" --time "00:15" --qos "debug" --mem-per-gpu 40G

test_boot:
	../exp72/simple_bootstrap.py 991 --tag ${TAG} --samples-plus-fpath ${PLUS_SAMPLES_FPATH_NU05} --samples-minus-fpath ${MINUS_SAMPLES_FPATH_NU05} --n-boots 5 --extra "nu50" --n-gals 1000


collate_interim:
# 	../exp70/collate_samples.py --tag ${TAG} --mode "plus" --start-string "interim_samples_${SEED}1" --contains "nu10"
# 	../exp70/collate_samples.py --tag ${TAG} --mode "minus" --start-string "interim_samples_${SEED}1" --contains "nu10"

# 	../exp70/collate_samples.py --tag ${TAG} --mode "plus" --start-string "interim_samples_${SEED}1" --contains "nu30"
# 	../exp70/collate_samples.py --tag ${TAG} --mode "minus" --start-string "interim_samples_${SEED}1" --contains "nu30"

# 	../exp70/collate_samples.py --tag ${TAG} --mode "plus" --start-string "interim_samples_${SEED}1" --contains "nu50"
# 	../exp70/collate_samples.py --tag ${TAG} --mode "minus" --start-string "interim_samples_${SEED}1" --contains "nu50"

# 	../exp70/collate_samples.py --tag ${TAG} --mode "plus" --start-string "interim_samples_${SEED}1" --contains "nu-20"
# 	../exp70/collate_samples.py --tag ${TAG} --mode "minus" --start-string "interim_samples_${SEED}1" --contains "nu-20"

# 	../exp70/collate_samples.py --tag ${TAG} --mode "plus" --start-string "interim_samples_${SEED}1" --contains "nu-40"
# 	../exp70/collate_samples.py --tag ${TAG} --mode "minus" --start-string "interim_samples_${SEED}1" --contains "nu-40"

# 	../exp70/collate_samples.py --tag ${TAG} --mode "plus" --start-string "interim_samples_${SEED}1" --contains "nu-60"
# 	../exp70/collate_samples.py --tag ${TAG} --mode "minus" --start-string "interim_samples_${SEED}1" --contains "nu-60"

# 	../exp70/collate_samples.py --tag ${TAG} --mode "plus" --start-string "interim_samples_${SEED}1" --contains "nu100"
# 	../exp70/collate_samples.py --tag ${TAG} --mode "minus" --start-string "interim_samples_${SEED}1" --contains "nu100"

# 	../exp70/collate_samples.py --tag ${TAG} --mode "plus" --start-string "interim_samples_${SEED}1" --contains "nu200"
# 	../exp70/collate_samples.py --tag ${TAG} --mode "minus" --start-string "interim_samples_${SEED}1" --contains "nu200"

# 	../exp70/collate_samples.py --tag ${TAG} --mode "plus" --start-string "interim_samples_${SEED}1" --contains "nu300"
# 	../exp70/collate_samples.py --tag ${TAG} --mode "minus" --start-string "interim_samples_${SEED}1" --contains "nu300"

######################################################

samples_nu:

# 	./slurm_get_interim_samples.py ${SEED}1 --tag ${TAG} --g1 ${SHEAR} --g2 0.0 --n-gals ${N_GALS} --n-samples-per-gal ${N_SAMPLES_PER_GAL} --time ${TIME_SAMPLE} --mode "plus" --mem-per-gpu ${MEM_PER_GPU} --qos ${QOS} --nodes ${NODES_SAMPLE} --nu 0.3

# 	./slurm_get_interim_samples.py ${SEED}1 --tag ${TAG} --g1 -${SHEAR} --g2 0.0 --n-gals ${N_GALS} --n-samples-per-gal ${N_SAMPLES_PER_GAL} --time ${TIME_SAMPLE} --mode "minus" --mem-per-gpu ${MEM_PER_GPU} --qos ${QOS} --nodes ${NODES_SAMPLE} --nu 0.3


# 	./slurm_get_interim_samples.py ${SEED}1 --tag ${TAG} --g1 ${SHEAR} --g2 0.0 --n-gals ${N_GALS} --n-samples-per-gal ${N_SAMPLES_PER_GAL} --time ${TIME_SAMPLE} --mode "plus" --mem-per-gpu ${MEM_PER_GPU} --qos ${QOS} --nodes ${NODES_SAMPLE} --nu 0.1

# 	./slurm_get_interim_samples.py ${SEED}1 --tag ${TAG} --g1 -${SHEAR} --g2 0.0 --n-gals ${N_GALS} --n-samples-per-gal ${N_SAMPLES_PER_GAL} --time ${TIME_SAMPLE} --mode "minus" --mem-per-gpu ${MEM_PER_GPU} --qos ${QOS} --nodes ${NODES_SAMPLE} --nu 0.1


# 	./slurm_get_interim_samples.py ${SEED}1 --tag ${TAG} --g1 ${SHEAR} --g2 0.0 --n-gals ${N_GALS} --n-samples-per-gal ${N_SAMPLES_PER_GAL} --time ${TIME_SAMPLE} --mode "plus" --mem-per-gpu ${MEM_PER_GPU} --qos ${QOS} --nodes ${NODES_SAMPLE} --nu 0.5

# 	./slurm_get_interim_samples.py ${SEED}1 --tag ${TAG} --g1 -${SHEAR} --g2 0.0 --n-gals ${N_GALS} --n-samples-per-gal ${N_SAMPLES_PER_GAL} --time ${TIME_SAMPLE} --mode "minus" --mem-per-gpu ${MEM_PER_GPU} --qos ${QOS} --nodes ${NODES_SAMPLE} --nu 0.5

# 	./slurm_get_interim_samples.py ${SEED}1 --tag ${TAG} --g1 ${SHEAR} --g2 0.0 --n-gals ${N_GALS} --n-samples-per-gal ${N_SAMPLES_PER_GAL} --time ${TIME_SAMPLE} --mode "plus" --mem-per-gpu ${MEM_PER_GPU} --qos ${QOS} --nodes ${NODES_SAMPLE} --nu 1.0

# 	./slurm_get_interim_samples.py ${SEED}1 --tag ${TAG} --g1 -${SHEAR} --g2 0.0 --n-gals ${N_GALS} --n-samples-per-gal ${N_SAMPLES_PER_GAL} --time ${TIME_SAMPLE} --mode "minus" --mem-per-gpu ${MEM_PER_GPU} --qos ${QOS} --nodes ${NODES_SAMPLE} --nu 1.0

# 	./slurm_get_interim_samples.py ${SEED}1 --tag ${TAG} --g1 ${SHEAR} --g2 0.0 --n-gals ${N_GALS} --n-samples-per-gal ${N_SAMPLES_PER_GAL} --time ${TIME_SAMPLE} --mode "plus" --mem-per-gpu ${MEM_PER_GPU} --qos ${QOS} --nodes ${NODES_SAMPLE} --nu 2.0

# 	./slurm_get_interim_samples.py ${SEED}1 --tag ${TAG} --g1 -${SHEAR} --g2 0.0 --n-gals ${N_GALS} --n-samples-per-gal ${N_SAMPLES_PER_GAL} --time ${TIME_SAMPLE} --mode "minus" --mem-per-gpu ${MEM_PER_GPU} --qos ${QOS} --nodes ${NODES_SAMPLE} --nu 2.0



# 	./slurm_get_interim_samples.py ${SEED}1 --tag ${TAG} --g1 ${SHEAR} --g2 0.0 --n-gals ${N_GALS} --n-samples-per-gal ${N_SAMPLES_PER_GAL} --time ${TIME_SAMPLE} --mode "plus" --mem-per-gpu ${MEM_PER_GPU} --qos ${QOS} --nodes ${NODES_SAMPLE} --nu -0.2

# 	./slurm_get_interim_samples.py ${SEED}1 --tag ${TAG} --g1 -${SHEAR} --g2 0.0 --n-gals ${N_GALS} --n-samples-per-gal ${N_SAMPLES_PER_GAL} --time ${TIME_SAMPLE} --mode "minus" --mem-per-gpu ${MEM_PER_GPU} --qos ${QOS} --nodes ${NODES_SAMPLE} --nu -0.2

# 	./slurm_get_interim_samples.py ${SEED}1 --tag ${TAG} --g1 ${SHEAR} --g2 0.0 --n-gals ${N_GALS} --n-samples-per-gal ${N_SAMPLES_PER_GAL} --time ${TIME_SAMPLE} --mode "plus" --mem-per-gpu ${MEM_PER_GPU} --qos ${QOS} --nodes ${NODES_SAMPLE} --nu -0.4

# 	./slurm_get_interim_samples.py ${SEED}1 --tag ${TAG} --g1 -${SHEAR} --g2 0.0 --n-gals ${N_GALS} --n-samples-per-gal ${N_SAMPLES_PER_GAL} --time ${TIME_SAMPLE} --mode "minus" --mem-per-gpu ${MEM_PER_GPU} --qos ${QOS} --nodes ${NODES_SAMPLE} --nu -0.4

	./slurm_get_interim_samples.py ${SEED}1 --tag ${TAG} --g1 ${SHEAR} --g2 0.0 --n-gals ${N_GALS} --n-samples-per-gal ${N_SAMPLES_PER_GAL} --time ${TIME_SAMPLE} --mode "plus" --mem-per-gpu ${MEM_PER_GPU} --qos ${QOS} --nodes ${NODES_SAMPLE} --nu 3.0

	./slurm_get_interim_samples.py ${SEED}1 --tag ${TAG} --g1 -${SHEAR} --g2 0.0 --n-gals ${N_GALS} --n-samples-per-gal ${N_SAMPLES_PER_GAL} --time ${TIME_SAMPLE} --mode "minus" --mem-per-gpu ${MEM_PER_GPU} --qos ${QOS} --nodes ${NODES_SAMPLE} --nu 3.0

	./slurm_get_interim_samples.py ${SEED}1 --tag ${TAG} --g1 ${SHEAR} --g2 0.0 --n-gals ${N_GALS} --n-samples-per-gal ${N_SAMPLES_PER_GAL} --time ${TIME_SAMPLE} --mode "plus" --mem-per-gpu ${MEM_PER_GPU} --qos ${QOS} --nodes ${NODES_SAMPLE} --nu -0.6

	./slurm_get_interim_samples.py ${SEED}1 --tag ${TAG} --g1 -${SHEAR} --g2 0.0 --n-gals ${N_GALS} --n-samples-per-gal ${N_SAMPLES_PER_GAL} --time ${TIME_SAMPLE} --mode "minus" --mem-per-gpu ${MEM_PER_GPU} --qos ${QOS} --nodes ${NODES_SAMPLE} --nu -0.6


test_slurm_samples:
	./slurm_get_interim_samples.py 991 --tag ${TAG} --g1 ${SHEAR} --g2 0.0 --n-gals 1000 --n-samples-per-gal 300 --mode "plus" --batch-size 1000 --qos "debug" --nodes 1 --time 00:10 --nu 0.1

test_samples_simple:
	./get_interim_samples.py 990 --tag ${TAG} --g1 ${SHEAR} --g2 0.0 --n-gals 100 --n-samples-per-gal 300 --mode "plus" --batch-size 100 --nu 0.1

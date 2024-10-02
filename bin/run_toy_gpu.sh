
export CUDA_VISIBLE_DEVICES="0"


# ./get_toy_ellip_samples.py --n-samples 10_000 --seed 39 --tag cpu_test1 --k 100 --obs-noise 1e-4 --shape-noise 1e-3 --> EASY dont need GPU

./get_shear_from_post_ellips.py --seed 401 --e-samples-file /pscratch/sd/i/imendoza/data/cache_chains/e_post_42_cpu_test2.npz

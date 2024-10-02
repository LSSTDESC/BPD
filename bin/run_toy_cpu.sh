
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS="cpu"

# easy, don't need GPU
# ./get_toy_ellip_samples.py --n-samples 10_000 --seed 42 --tag cpu_test2 --k 10 --obs-noise 1e-4 --shape-noise 1e-3

# much harder in general, likelihood might be much faster to compute in GPU? 
./get_shear_from_post_ellips.py --seed 39 --e-samples-file /pscratch/sd/i/imendoza/data/cache_chains/e_post_42_cpu_test2.npz

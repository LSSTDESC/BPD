
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS="cpu"

# easy, don't need GPU
# ./get_toy_ellip_samples.py --n-samples 10_000 --seed 42 --tag cpu_test2 --k 10 --obs-noise 1e-4 --shape-noise 1e-3

# much harder in general, likelihood might be much faster to compute in GPU? 
# better if k is smaller than 100
# ./get_shear_from_post_ellips.py --seed 39 --e-samples-file /pscratch/sd/i/imendoza/data/cache_chains/e_post_42_cpu_test2.npz

# for i in {3..100}
# do
#   echo $i
#   ./get_toy_ellip_samples.py --n-samples 10_000 --seed $i --tag experiment_cpu1 --k 10 --obs-noise 1e-4 --shape-noise 1e-3
# done

for i in {55..100}
do
  echo $i
  ./get_shear_from_post_ellips.py --seed $i --e-samples-file /pscratch/sd/i/imendoza/data/cache_chains/experiment_cpu1/e_post_${i}_experiment_cpu1.npz
done

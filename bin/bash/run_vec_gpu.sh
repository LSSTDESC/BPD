export CUDA_VISIBLE_DEVICES="1"
echo "Using CUDA ${CUDA_VISIBLE_DEVICES}"

./vect_toy_shear_gpu.py --n-samples-gals 1000 --n-samples-shear 3000 --n-vec 100 --start-seed 1 --end-seed 1000 --tag "gpu3_vec" --k 200 --trim 1

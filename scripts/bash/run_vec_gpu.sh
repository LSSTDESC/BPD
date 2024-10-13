export CUDA_VISIBLE_DEVICES=$1
echo "Using CUDA ${CUDA_VISIBLE_DEVICES}"
START=$2
END=$3

./vect_toy_shear_gpu.py --n-samples-gals 1_000 --n-samples-shear 3000 --n-vec 100 --start-seed $START --end-seed $END --tag "gpu3_vec" --k 100 --trim 1


export CUDA_VISIBLE_DEVICES=$1
echo "Using CUDA ${CUDA_VISIBLE_DEVICES}"
export JAX_ENABLE_X64="False"
START=$2
END=$3

for ((i = $START; i <= $END; i++)); do
    echo $i
    ./get_toy_ellip_samples.py --tag gpu2 --seed $i --k 50
    ./get_shear_from_post_ellips.py --tag gpu2 --seed $i --e-samples-fname e_post_${i}.npz
done

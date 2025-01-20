export CUDA_VISIBLE_DEVICES="0"
export JAX_ENABLE_X64="True"
SEED="42"
 
../../scripts/get_shear_from_shapes.py $SEED --old-seed $SEED --interim-samples-fname "interim_samples_${SEED}_plus.npz" --tag exp40_$SEED --overwrite --extra-tag "plus"
../../scripts/get_shear_from_shapes.py $SEED --old-seed $SEED --interim-samples-fname "interim_samples_${SEED}_minus.npz" --tag exp40_$SEED --overwrite --extra-tag "minus"
# ./get_shear_jackknife.py $SEED --old-seed $SEED --samples-plus-fname "interim_samples_${SEED}_plus.npz" --samples-minus-fname "interim_samples_${SEED}_minus.npz" --tag exp40_$SEED --overwrite --n-jacks 50

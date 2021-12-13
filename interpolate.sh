source common.sh

name_exp_test=${name_exp}_interpolate
path_ckpt=logs/$name_exp/checkpoints
path_latest=$(find $path_ckpt -name *.pth | sort | tail -n 1)

#rm logs/$name_exp_test -rfv

CUDA_VISIBLE_DEVICES=1 python experiment_scripts/interpolate.py \
                              --checkpoint_path=$path_latest \
                              --experiment_name=$name_exp_test \
                              --resolution=512 \
                              --dim_embd=$dim_embd \
                              --num_class=$num_class \
                              --dim_hidden=$dim_hidden \
                              --num_layer=$num_layer  \
                              --shapes $shape1 $shape2

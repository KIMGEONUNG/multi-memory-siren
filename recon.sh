source common.sh

# name_exp='experiment_5'

name_exp_test=${name_exp}_recon
path_ckpt=logs/$name_exp/checkpoints
path_latest=$(find $path_ckpt -name *.pth | sort | tail -n 1)
path_target='xyzs_10000_test/cbe006da89cca7ffd6bab114dd47e3f.xyzn'
size_batch=2000
num_iter=500

CUDA_VISIBLE_DEVICES=1 python experiment_scripts/recon_sdf.py \
                              --checkpoint_path=$path_latest \
                              --batch_size=$size_batch \
                              --experiment_name=$name_exp_test \
                              --resolution=512 \
                              --dim_embd=$dim_embd \
                              --num_class=$num_class \
                              --num_iter=$num_iter\
                              --dim_hidden=$dim_hidden \
                              --num_layer=$num_layer \
                              --dropout=$dropout \
                              --path_target=$path_target 

source common.sh

# xyz=./xyzs_old1
xyz='xyznrgb_test'
rm -rf logs//$name_exp

size_batch=160000
size_batch=1600

CUDA_VISIBLE_DEVICES=1 python experiment_scripts/train_sdfc.py \
       --model_type=sine \
       --point_cloud_path=$xyz \
       --batch_size=$size_batch \
       --experiment_name=$name_exp \
       --dim_embd=$dim_embd \
       --num_class=$num_class \
       --dim_hidden=$dim_hidden \
       --num_layer=$num_layer \
       --dropout=$dropout 

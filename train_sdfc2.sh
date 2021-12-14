source common.sh

xyz='xyznrgb'
size_batch=10000

CUDA_VISIBLE_DEVICES=3 python experiment_scripts/train_sdfc.py \
       --model_type=sine \
       --point_cloud_path=$xyz \
       --batch_size=$size_batch \
       --experiment_name=$name_exp \
       --dim_embd=$dim_embd \
       --dim_embdc=$dim_embdc \
       --c2_conditioned=$conditioned  \
       --num_class=$num_class \
       --dim_hidden=$dim_hidden \
       --num_layer=$num_layer \
       --dropout=$dropout 

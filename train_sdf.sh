source common.sh

xyz=./xyzs_10000
# xyz=./xyzs_old1
size_batch=160000

python experiment_scripts/train_sdf.py \
       --model_type=sine \
       --point_cloud_path=$xyz \
       --batch_size=$size_batch \
       --experiment_name=$name_exp \
       --dim_embd=$dim_embd \
       --num_class=$num_class \
       --dim_hidden=$dim_hidden \
       --num_layer=$num_layer \
       --dropout=$dropout 
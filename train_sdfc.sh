source common.sh

xyz='xyznrgb_dense2'
size_batch=120000

CUDA_VISIBLE_DEVICES=0 python experiment_scripts/train_sdfc.py \
       --model_type=sine \
       --point_cloud_path=$xyz \
       --batch_size=$size_batch \
       --experiment_name=$name_exp \
       --dim_embd=$dim_embd \
       --num_class=$num_class \
       --dim_hidden=$dim_hidden \
       --num_layer=$num_layer \
       --dropout=$dropout 

cp common.sh logs/$name_exp_test
cp test_sdfc.sh logs/$name_exp_test

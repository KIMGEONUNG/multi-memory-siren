# Reconstruction for Shapes and Textures with SIREN

- Team.4DGraphics
- KIMGEONUNG & SONJOOEUN

## Environment 

We provide conda environment for running code

```
conda env create --file environment.yaml
```


## Train 

You can simply run the predefined training script.

```
bash train_sdfc.sh
```

You can also run the training code with a variaty of arguments.
```
python experiment_scripts/train_sdfc.py \
       --model_type=sine \  # sine means SIREN network
       --point_cloud_path=[dataset_path] \
       --batch_size=[batch_size] \
       --experiment_name=[log_directory_name] \
       --dim_embd=[shape_and_color_embedding_dimension] \
       --num_class=[the_number_of_classes] \
       --dim_hidden=[MLP_width] \
       --num_layer=[MLP_depth] \
       --dropout=[probability_of_dropout]
```

Our code does not support multi-GPU envorinments.


## Test

You can simply run the predefined test script.

```
bash train_sdfc.sh
```

You can also run the test code with a variaty of arguments.

```
python experiment_scripts/test_sdfc.py \
      --checkpoint_path=$path_latest \
      --experiment_name=[output_directory_name] \
      --resolution=512 \
      --dim_embd=[shape_and_color_embedding_dimension] \
      --num_class=[the_number_of_classes] \
      --dim_hidden=[MLP_width] \
      --num_layer=[MLP_depth] \
      --dropout=[probability_of_dropout]
  
```

>**Note that**
>- For dim_embd, num_class, dim_hidden, num_layer, dropout,
>You must use same arguments which was used for training.
>- common.sh file has common argument info for train and test.
>- Dataset must have 9 dimension data which is x,y,z coordinates, normal and RGB 


## Dataset 

We basically use ShapeNetCore.v2 dataset with custom preprocessing. The 
preprocessing includes point sampling, normal estimation and rgb estimation.
You can download the dataset to the following link. This dataset is XYZNRGB data
to chair class in ShapeNetCor.v2.
- [Download Dataset](https://drive.google.com/drive/folders/1HIpmxAxKM0XBJ1dUrXZLGLdKmi_4obKE?usp=sharing)


## Reference

Our code was referenced a lot below.  

- [SIREN](https://github.com/vsitzmann/siren) (NIPS 2020)
- [DeepSDF](https://github.com/facebookresearch/DeepSDF) (CVPR2019)

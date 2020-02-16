#!/usr/bin/env bash

OUTPUT_DIR=$1
MODE=$2
EPOCHS=$3
LR=$4
CKPT_LOAD_MODE=$5
DROPOUT_KEEP_PROB=$6
D_PICKLE_NAME=$7
VALID_PICKLE_NAME=$8
U_PICKLE_NAME=$9
GAMMA=${10}
LAMDA=${11}
USE_JOINT_f_w=${12}

DATA_DIR=../../data/CENSUS



W_LAYERS="256,256"
F_LAYERS="256,256"

F_D_CLASS_SAMPLING=5,5 # while mixing d and U sets, oversample data from d
                                     # this is because size of d is much smaller
                                     # than size of U
                                     # this is done so that in any batch there are enough instances from "d"
                                     # along with instances from U

python3.6 -u main.py \
  --output_dir="$DATA_DIR"/outputs/"$OUTPUT_DIR" \
  --run_mode=$MODE \
  --checkpoint_load_mode=$CKPT_LOAD_MODE \
  --data_dir=$DATA_DIR \
  --f_d_primary_metric=accuracy \
  --f_d_epochs=$EPOCHS \
  --f_d_U_epochs=$EPOCHS \
  --f_d_batch_size=16 \
  --f_d_U_batch_size=64 \
  --f_d_adam_lr=$LR \
  --f_d_U_adam_lr=$LR \
  --validation_pickle_name=$VALID_PICKLE_NAME \
  --d_pickle_name=$D_PICKLE_NAME \
  --dropout_keep_prob=$DROPOUT_KEEP_PROB \
  --w_layers_str=$W_LAYERS \
  --f_layers_str=$F_LAYERS \
  --f_d_class_sampling_str=$F_D_CLASS_SAMPLING \
  --U_pickle_name=$U_PICKLE_NAME \
  --gamma=$GAMMA \
  --lamda=$LAMDA \
  --early_stopping_p=20 \
  --use_joint_f_w=$USE_JOINT_f_w

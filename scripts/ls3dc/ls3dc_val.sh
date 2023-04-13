#!/bin/bash
export OMP_NUM_THREADS=3

CURR_DBSCAN=14.0
CURR_TOPK=750
CURR_QUERY=160
CURR_SIZE=6

# TRAIN
python main_instance_segmentation.py \
general.experiment_name="validation_ls3dc" \
general.project_name="ls3dc" \
data=primitives \
data/datasets=ls3dc \
general.num_targets=3 \
data.num_labels=3 \
data.voxel_size=0.04 \
data.num_workers=1 \
data.cache_data=true \
data.cropping_v1=false \
general.reps_per_epoch=100 \
model.num_queries=${CURR_QUERY} \
general.on_crops=true \
model.config.backbone._target_=models.Res16UNet18B \
data.crop_length=${CURR_SIZE} \
general.eval_inner_core=6.0

# TEST
python main_instance_segmentation.py \
general.experiment_name="validation_ls3dc_query_${CURR_QUERY}_topk_${CURR_TOPK}_dbscan_${CURR_DBSCAN}_size_${CURR_SIZE}" \
general.project_name="ls3dc_eval" \
data/datasets=ls3dc \
data=primitives \
general.num_targets=3 \
data.num_labels=3 \
data.voxel_size=0.04 \
data.num_workers=10 \
data.cache_data=true \
data.cropping_v1=false \
general.reps_per_epoch=100 \
model.num_queries=${CURR_QUERY} \
general.on_crops=true \
model.config.backbone._target_=models.Res16UNet18B \
general.train_mode=false \
general.checkpoint="checkpoints/ls3dc/ls3dc_val.ckpt" \
data.crop_length=${CURR_SIZE} \
general.eval_inner_core=50.0 \
general.topk_per_image=${CURR_TOPK} \
general.use_dbscan=true \
general.dbscan_eps=${CURR_DBSCAN}

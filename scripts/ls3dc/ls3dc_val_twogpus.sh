#!/bin/bash
export OMP_NUM_THREADS=3

CURR_DBSCAN=14.0
CURR_TOPK=750
CURR_QUERY=160
CURR_SIZE=6

# TRAIN
python main_instance_segmentation.py \
general.experiment_name="validation_ls3dc_test" \
general.project_name="ls3dc_test" \
data=primitives \
data/datasets=ls3dc \
general.num_targets=3 \
data.num_labels=3 \
data.voxel_size=0.04 \
data.num_workers=10 \
data.cache_data=true \
data.cropping_v1=false \
general.reps_per_epoch=1 \
model.num_queries=${CURR_QUERY} \
general.on_crops=false \
model.config.backbone._target_=models.Res16UNet18B \

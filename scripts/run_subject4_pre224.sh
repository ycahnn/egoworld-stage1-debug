#!/usr/bin/env bash
set -euo pipefail

cd /home/youngchan/egoworld/egoworld-stage1-debug

mkdir -p logs data/preprocessed_224_png_nvme/subject4
log_path="logs/subject4_pre224_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$log_path") 2>&1

echo "Subject4 pre224 preprocessing started at $(date)"
echo "Log: $log_path"

common_args=(
  --out_root data/preprocessed_224_png_nvme/subject4
  --image_root /mnt/hdd2/youngchan/h2o_dataset
  --label_root /mnt/hdd2/youngchan/h2o_dataset
  --workers 8
)

python vit_handpose/preprocess_h2o_pre224.py \
  --src_csv data/h2o_subject4_cam0123_to_cam4_hand_pose_train.csv \
  --output_csv data/preprocessed_224_png_nvme/subject4/h2o_subject4_cam0123_to_cam4_hand_pose_train_pre224.csv \
  --image_subdir train \
  "${common_args[@]}"

python vit_handpose/preprocess_h2o_pre224.py \
  --src_csv data/h2o_subject4_cam0123_to_cam4_hand_pose_val.csv \
  --output_csv data/preprocessed_224_png_nvme/subject4/h2o_subject4_cam0123_to_cam4_hand_pose_val_pre224.csv \
  --image_subdir val \
  "${common_args[@]}"

python vit_handpose/preprocess_h2o_pre224.py \
  --src_csv data/h2o_subject4_cam0123_to_cam4_hand_pose_test.csv \
  --output_csv data/preprocessed_224_png_nvme/subject4/h2o_subject4_cam0123_to_cam4_hand_pose_test_pre224.csv \
  --image_subdir test \
  "${common_args[@]}"

echo "Subject4 pre224 preprocessing finished at $(date)"

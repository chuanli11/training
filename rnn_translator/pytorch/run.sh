#!/bin/bash

set -e

DATASET_DIR='/data'

SEED=${1:-"1"}
TARGET=${2:-"24.00"}

# Single-GPU training
python3 train.py \
  --dataset-dir ${DATASET_DIR} \
  --seed $SEED \
  --target-bleu $TARGET \
  --train-batch-size 64 \
  --val-batch-size 32 \
  --test-batch-size 32 

# Multi-GPU training
# python3 -m torch.distributed.launch --nproc_per_node=2 train.py \
#   --dataset-dir ${DATASET_DIR} \
#   --seed $SEED \
#   --target-bleu $TARGET \
#   --train-batch-size 64 \
#   --val-batch-size 32 \
#   --test-batch-size 32

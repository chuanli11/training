#/bin/bash

RANDOM_SEED=$1
QUALITY=$2
set -e

# Register the model as a source root
export PYTHONPATH="$(pwd):${PYTHONPATH}"

MODEL_DIR="/tmp/resnet_imagenet_${RANDOM_SEED}"

# python3 official/resnet/imagenet_main.py $RANDOM_SEED --data_dir /imn/imagenet/combined/  \
#   --model_dir $MODEL_DIR --train_epochs 10000 --stop_threshold $QUALITY --batch_size 64 \
#   --version 1 --resnet_size 50 --epochs_between_evals 4


python3 official/resnet/imagenet_main.py $RANDOM_SEED --data_dir /imn/imagenet/combined/  \
  --model_dir $MODEL_DIR --train_epochs 10000 --stop_threshold $QUALITY --batch_size 64 \
  --version 1 --resnet_size 50 --epochs_between_evals 4 --num_gpus 2 --dtype fp16

# To run multiple, instead run:
# Scale the batch size by num_gpus
# python3 official/resnet/imagenet_main.py $RANDOM_SEED --data_dir /imn/imagenet/combined/ \
#   --model_dir $MODEL_DIR --train_epochs 2 --stop_threshold $QUALITY --batch_size 64 \
#   --version 1 --resnet_size 50 --dtype fp16 --num_gpus 2 \
#   --epochs_between_evals 4

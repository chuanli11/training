#!/bin/bash

set -e

SEED=$1
QUALITY=$2

cd /research/transformer

export PYTHONPATH=/research/transformer/transformer:${PYTHONPATH}
# Add compliance to PYTHONPATH
# export PYTHONPATH=/mlperf/training/compliance:${PYTHONPATH}

## Base model
python3 transformer/transformer_main.py --random_seed=${SEED} --data_dir=processed_data/ --model_dir=model --params=base --bleu_threshold ${QUALITY} --bleu_source=newstest2014.en --bleu_ref=newstest2014.de

## Big model needs extra VRAM
# python3 transformer/transformer_main.py --random_seed=${SEED} --data_dir=processed_data/ --model_dir=model --params=big --bleu_threshold ${QUALITY} --bleu_source=newstest2014.en --bleu_ref=newstest2014.de

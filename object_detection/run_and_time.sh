#!/bin/bash

# Runs benchmark and reports time to convergence

pushd pytorch

# Single GPU training
# time python tools/train_mlperf.py --config-file "configs/e2e_mask_rcnn_R_50_FPN_1x.yaml" \
#        SOLVER.IMS_PER_BATCH 2 TEST.IMS_PER_BATCH 1 SOLVER.MAX_ITER 720000 SOLVER.STEPS "(480000, 640000)" SOLVER.BASE_LR 0.0025

# Multi-GPU training
time python -m torch.distributed.launch --nproc_per_node=2 tools/train_mlperf.py --config-file "configs/e2e_mask_rcnn_R_50_FPN_1x.yaml" \
       SOLVER.IMS_PER_BATCH 2 TEST.IMS_PER_BATCH 1 SOLVER.MAX_ITER 720000 SOLVER.STEPS "(480000, 640000)" SOLVER.BASE_LR 0.0025


popd

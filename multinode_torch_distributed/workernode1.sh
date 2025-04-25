#!/bin/bash
export MASTER_ADDR=on-demand-testing-rodri4-1
export MASTER_PORT=29500
export NNODES=2
export GPUS_PER_NODE=8
export NODE_RANK=0

torchrun \
  --nnodes=$NNODES \
  --nproc_per_node=$GPUS_PER_NODE \
  --node_rank=$NODE_RANK \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  multinode_torch_distributed.py


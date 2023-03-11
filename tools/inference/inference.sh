#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
IMG0=$3
IMG1=$4
IMG2=$5
SAVE_DIR=$6
NUM=$7
PORT=${PORT:-29500}

# inference with three imgs
PYTHONPATH="$(dirname $0)/../..":$PYTHONPATH \
  python -m torch.distributed.launch --nproc_per_node=1 --master_port=$PORT \
  $(dirname "$0")/inference.py --config $CONFIG --checkpoint $CHECKPOINT --img_pre $IMG0 --img_cur $IMG1 --img_nxt $IMG2 --save_dir $SAVE_DIR --num $NUM

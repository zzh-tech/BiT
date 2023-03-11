#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
SAVE_DIR=$3
DATA_DIR=$4
PORT=${PORT:-29501}

# calculate metrics and save result images
PYTHONPATH="$(dirname $0)/../..":$PYTHONPATH \
  python -m torch.distributed.launch --nproc_per_node=1 --master_port=$PORT \
  $(dirname "$0")/test_bit_adobe240.py --config $CONFIG --checkpoint $CHECKPOINT --save_dir $SAVE_DIR --data_dir $DATA_DIR

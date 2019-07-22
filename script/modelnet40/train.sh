#!/usr/bin/env bash

## run the training
python train.py \
--datasets datasets/modelnet40 \
--name modelnet \
--batch_size 32 \
--nclasses 40
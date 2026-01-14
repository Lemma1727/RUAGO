#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# ViT & CIFAR-100
python src/main.py \
  --data_name cifar100 \
  --model_name ViT \
  --lr 1e-5 \
  --gamma_1 0.1 \
  --gamma_2 0.01 \
  --eps 32 \
  --alpha 0.1 \
  --iters 3 \
  --epoch 2 \
  --batch_size 64 \
  --synthesis_batch_size 32 \
  "$@"



#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Pretrain (retrain baseline) ResNet18 & CIFAR-10
python src/pretrain.py \
  --data_name cifar10 \
  --model_name ResNet18 \
  --mode retrain \
  "$@"



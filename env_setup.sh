#!/bin/bash
set -euo pipefail
module purge >/dev/null 2>&1 || true
module load profile/deeplrn >/dev/null 2>&1 || true
module load python/3.11.7 >/dev/null 2>&1 || true
module load cuda/12.6 >/dev/null 2>&1 || true
export CIFAR100_ROOT=/leonardo_work/IscrC_YENDRI/paerle/Cifar100Speedrun
export PIP_CACHE_DIR=/leonardo_work/IscrC_YENDRI/paerle/.cache/pip
export TORCH_HOME=/leonardo_work/IscrC_YENDRI/paerle/.cache/torch
export TORCHINDUCTOR_CACHE_DIR=/leonardo_work/IscrC_YENDRI/paerle/.cache/torchinductor
export TRITON_CACHE_DIR=/leonardo_work/IscrC_YENDRI/paerle/.cache/triton
source /leonardo_work/IscrC_YENDRI/paerle/CIfar10Speedrun/cifar10-speedrun/.venv/bin/activate
cd "$CIFAR100_ROOT/cifar100-benchmark"
echo "[env_setup] cifar100 ready (python=$(python --version))"

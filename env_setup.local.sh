#!/bin/bash
set -euo pipefail
# Per-user (dcrisost) equivalent of env_setup.sh. Semantics preserved: same modules, same
# shared cifar10-speedrun venv (it is readable). Differences, all for portability/writability:
#   - CIFAR100_ROOT derived from THIS file's location (works from any checkout)
#   - all caches redirected to my own $SCRATCH (paerle's cache dirs are read-only to me)
# No proxy: like paerle's original, the CIFAR-100 .pt tensors are pre-staged into
# cifar100-benchmark/cifar100/, so prepare_cifar100_hf.py runs fully offline (no download).
module purge >/dev/null 2>&1 || true
module load profile/deeplrn >/dev/null 2>&1 || true
module load python/3.11.7 >/dev/null 2>&1 || true
module load cuda/12.6 >/dev/null 2>&1 || true

export CIFAR100_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# caches on my own scratch (writable, auto-purged)
CACHE="${SCRATCH:-/leonardo_scratch/large/userexternal/$USER}/cifar100_cache"
export PIP_CACHE_DIR="$CACHE/pip"
export TORCH_HOME="$CACHE/torch"
export TORCHINDUCTOR_CACHE_DIR="$CACHE/torchinductor"
export TRITON_CACHE_DIR="$CACHE/triton"
export HF_HOME="$CACHE/huggingface"
mkdir -p "$PIP_CACHE_DIR" "$TORCH_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$HF_HOME"

# reuse the shared venv (readable) — do NOT rebuild
source /leonardo_work/IscrC_YENDRI/paerle/CIfar10Speedrun/cifar10-speedrun/.venv/bin/activate
cd "$CIFAR100_ROOT/cifar100-benchmark"

# Stage the CIFAR-100 tensors from the shared (paerle) checkout if missing, so
# prepare_cifar100_hf.py runs fully offline (compute nodes have no internet here, and the
# repo's cifar100/ is gitignored → a fresh checkout has none). Done at job start so there is
# no window for the files to be removed before use.
_shared=/leonardo_work/IscrC_YENDRI/paerle/Cifar100Speedrun/cifar100-benchmark/cifar100
mkdir -p cifar100
for _f in train.pt test.pt; do
  [ -s "cifar100/$_f" ] || cp "$_shared/$_f" "cifar100/$_f"
done

echo "[env_setup.local] cifar100 ready (root=$CIFAR100_ROOT python=$(python --version 2>&1) data=$(ls cifar100/*.pt 2>/dev/null | wc -l) pt-files)"

#!/bin/bash
# Per-user (dcrisost) experiment launcher. Runs the additive train_cifar100_resnet_muon_exp.py
# from MY fork checkout, writes logs to MY space, and sources env_setup.local.sh (own caches +
# shared venv + offline data self-staging). Modeled on smoke.local.sh (header/guard/env) and
# official_baseline.sh (runs/epochs defaults + analyze). Additive; the paerle-path originals are
# untouched. All recipe knobs (C100_PRECISION, C100_LABEL_SMOOTHING, C100_SCHEDULE, ...) are NOT
# set here — leo exports them on the submit line and they pass through untouched to python.
#SBATCH --job-name=c100-exp-dcrisost
#SBATCH --account=IscrC_SIMP
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH -D /leonardo_work/IscrC_TVU/dcrisost/cifar100-speedrun
#SBATCH --output=./logs/exp-%j.out
#SBATCH --error=./logs/exp-%j.err
set -euo pipefail
if [[ "${SLURM_JOB_ACCOUNT:-}" != "iscrc_simp" && "${SLURM_JOB_ACCOUNT:-}" != "IscrC_SIMP" ]]; then
  echo "Refusing to run outside IscrC_SIMP." >&2
  exit 2
fi
cd /leonardo_work/IscrC_TVU/dcrisost/cifar100-speedrun
# env_setup.local.sh exports CIFAR100_ROOT (this repo root) and cd's into cifar100-benchmark.
source env_setup.local.sh
echo "==> $(date) job=${SLURM_JOB_ID:-N/A} node=$(hostname)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
python prepare_cifar100_hf.py
# Only RUNS/EPOCHS/TARGET/SLEEP get script-side defaults; every C100_* recipe knob is inherited
# untouched from the submit environment. SLEEP defaults to 1e9 cycles for real runs (as in
# official_baseline) but is overridable.
C100_RUNS=${C100_RUNS:-30} \
C100_EPOCHS=${C100_EPOCHS:-16} \
C100_TARGET=${C100_TARGET:-0.70} \
C100_SLEEP_CYCLES=${C100_SLEEP_CYCLES:-1000000000} \
  python train_cifar100_resnet_muon_exp.py
# SLURM writes the log under the -D repo root (logs/), but env_setup.local.sh left us in the
# cifar100-benchmark subdir, so reference the log via CIFAR100_ROOT rather than a bare ./logs.
python analyze_cifar100.py "${CIFAR100_ROOT}/logs/exp-${SLURM_JOB_ID}.out" --target "${C100_TARGET:-0.70}" || true
echo "==> exp done $(date)"

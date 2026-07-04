#!/bin/bash
# Per-user (dcrisost) tranche-1 G2 screen: champion + 8 candidate arms, one arm =
# one `python train_..._exp.py` invocation, all in ONE job so every candidate is
# paired same-node/same-seeds against the champion arm (controls cross-node time
# variance). Each arm prints `>>> ARM <name>` then its own config/table; parse the
# combined log with scratchpad/parse_tranche.py. Screening only: 5 seeds @ 7ep,
# SLEEP_CYCLES=0. NOT a record run.
#SBATCH --job-name=c100-tr1-dcrisost
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
#SBATCH --output=./logs/tr1-%j.out
#SBATCH --error=./logs/tr1-%j.err
set -uo pipefail
if [[ "${SLURM_JOB_ACCOUNT:-}" != "iscrc_simp" && "${SLURM_JOB_ACCOUNT:-}" != "IscrC_SIMP" ]]; then
  echo "Refusing to run outside IscrC_SIMP." >&2
  exit 2
fi
cd /leonardo_work/IscrC_TVU/dcrisost/cifar100-speedrun
source env_setup.local.sh
echo "==> $(date) job=${SLURM_JOB_ID:-N/A} node=$(hostname)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
python prepare_cifar100_hf.py

# Base = champion-frontier recipe @ 7 epochs, 5-seed screen, no inter-run sleep.
export C100_RUNS=5 C100_EPOCHS=7 C100_TARGET=0.70 C100_SLEEP_CYCLES=0 C100_BATCH=1024 C100_COMPILE=1
export C100_PRECISION=bf16 C100_LABEL_SMOOTHING=0.15 C100_SCHEDULE=wsd
export C100_WARMUP_FRAC=0.05 C100_COOLDOWN_FRAC=0.40
export C100_MUON_NESTEROV=1 C100_NS_STEPS=3 C100_MUON_MOM_WARMUP=0.15 C100_MUON_MOM_START=0.85
export C100_CROP_PAD=3 C100_MUON_LR=0.055

# One arm = champion base env plus (optionally) a single delta passed via `env`.
# `|| echo FAILED` keeps one bad arm from aborting the batch; each `env VAR=..`
# override is isolated to that invocation (module-level knobs like C100_RESNET_D /
# C100_PRECISION are read at import, so they must be set on the python process).
run_arm() {
  name="$1"; shift
  echo ">>> ARM ${name}"
  env "$@" python train_cifar100_resnet_muon_exp.py || echo ">>> ARM ${name} FAILED rc=$?"
}

run_arm champion
run_arm head_adam   C100_HEAD_OPT=adam C100_HEAD_LR=0.003
run_arm bnbeta8     C100_BN_SHIFT_LR_MULT=8
run_arm ema99       C100_EMA_DECAY=0.99
run_arm lookahead   C100_LOOKAHEAD_K=5 C100_LOOKAHEAD_ALPHA=0.5
run_arm ghostbn32   C100_GHOST_BN=32
run_arm dirac       C100_DIRAC_INIT=1
run_arm resnetd     C100_RESNET_D=1
run_arm presize24   C100_RESIZE_PX=24 C100_RESIZE_FRAC=0.5

echo "==> tranche1 done $(date)"

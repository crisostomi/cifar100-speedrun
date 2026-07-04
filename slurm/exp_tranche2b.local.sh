#!/bin/bash
# Tranche-2b code-lever screen: firm the fragile 6-epoch floor with the round-3
# knobs. Base = s3p stack @6ep (current 6ep best, 0.7016 mean 22/30). Each arm adds
# one code-lever delta. Includes the vectorized Ghost-BN RESURRECTION (+0.29pp acc,
# was killed only on the old impl's wall-time). 8-seed paired screen, sleep 0.
#SBATCH --job-name=c100-tr2b-dcrisost
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
#SBATCH --output=./logs/tr2b-%j.out
#SBATCH --error=./logs/tr2b-%j.err
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

# Base = full s3p stack @ 6 epochs.
export C100_RUNS=8 C100_EPOCHS=6 C100_TARGET=0.70 C100_SLEEP_CYCLES=0 C100_BATCH=1024 C100_COMPILE=1
export C100_PRECISION=bf16 C100_LABEL_SMOOTHING=0.15 C100_SCHEDULE=wsd
export C100_WARMUP_FRAC=0.05 C100_COOLDOWN_FRAC=0.40
export C100_MUON_NESTEROV=1 C100_NS_STEPS=3 C100_MUON_MOM_WARMUP=0.15 C100_MUON_MOM_START=0.85
export C100_CROP_PAD=3 C100_MUON_LR=0.055
export C100_RESNET_D=1 C100_LOOKAHEAD_K=5 C100_BN_SHIFT_LR_MULT=8 C100_RESIZE_PX=24 C100_RESIZE_FRAC=0.5

run_arm() {
  name="$1"; shift
  echo ">>> ARM ${name}"
  env "$@" python train_cifar100_resnet_muon_exp.py || echo ">>> ARM ${name} FAILED rc=$?"
}

run_arm s3p6_ref
run_arm ghostbn32_6  C100_GHOST_BN=32
run_arm rezero02_6   C100_RESIDUAL_ALPHA=0.2
run_arm agc016_6     C100_AGC=0.16
run_arm blurpool_6   C100_BLURPOOL=1
run_arm detflip_6    C100_DET_FLIP=1
run_arm lsanneal_6   C100_LS_FINAL=0.05
run_arm stem2s2_6    C100_STEM=conv2s2 C100_RESIZE_PX=0 C100_RESIZE_FRAC=0.0

echo "==> tranche2b done $(date)"

#!/bin/bash
# 7ep record speed push. The E2 record (s3p_7, 9.487s, 0.7132, 30/30) has a large
# +1.32pp margin above 70%. Spend it on speed via more aggressive progressive resize
# (smaller px and/or larger low-res fraction), which is a free train-only lever. Any
# arm that stays robustly >70% at less time is a new record candidate. Base = s3p_7
# @7ep; resize variants. 8-seed paired screen, sleep 0.
#SBATCH --job-name=c100-7spd-dcrisost
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
#SBATCH --output=./logs/7spd-%j.out
#SBATCH --error=./logs/7spd-%j.err
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

export C100_RUNS=8 C100_EPOCHS=7 C100_TARGET=0.70 C100_SLEEP_CYCLES=0 C100_BATCH=1024 C100_COMPILE=1
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

run_arm s3p_7_ref
run_arm resize20f5_7   C100_RESIZE_PX=20
run_arm resize16f5_7   C100_RESIZE_PX=16
run_arm resize24f7_7   C100_RESIZE_FRAC=0.7
run_arm resize20f7_7   C100_RESIZE_PX=20 C100_RESIZE_FRAC=0.7

echo "==> 7ep speed push done $(date)"

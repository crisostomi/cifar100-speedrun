#!/bin/bash
# Tranche-2a env-only screen: firm up the FRAGILE 6-epoch floor break. Base = the
# current 6ep best (s3p_6 = champ + ResNet-D + Lookahead + BN-beta + resize24 @6ep,
# screened 0.7023 mean, 6/8). Each arm adds one env-only delta aiming to lift the
# 6ep mean/min toward a clean 30/30. Paired same-node, 8 seeds, screening (sleep 0).
#SBATCH --job-name=c100-tr2a-dcrisost
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
#SBATCH --output=./logs/tr2a-%j.out
#SBATCH --error=./logs/tr2a-%j.err
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

# Base = full s3p stack @ 6 epochs (ResNet-D read at import, so exported here).
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
run_arm headsgd_6     C100_HEAD_OPT=sgd
run_arm headadam01_6  C100_HEAD_OPT=adam C100_HEAD_LR=0.01
run_arm ema90_6       C100_EMA_DECAY=0.9
run_arm wide_6        C100_WIDTHS=96,192,384
run_arm muonlr06_6    C100_MUON_LR=0.06
run_arm cooldown50_6  C100_COOLDOWN_FRAC=0.5

echo "==> tranche2a done $(date)"

#!/bin/bash
# 5-epoch frontier probe. tranche-2a showed 6ep is partly CAPACITY-limited (wide net
# +2.23pp, 8/8). Test whether capacity buys enough margin to reach 70% at 5 epochs
# (~6.8s) -- a further record below the 6ep floor. Base = s3p stack @5ep; width and
# hotter-LR variants. 8-seed paired screen, sleep 0.
#SBATCH --job-name=c100-5ep-dcrisost
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
#SBATCH --output=./logs/5ep-%j.out
#SBATCH --error=./logs/5ep-%j.err
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

# Base = s3p stack @ 5 epochs.
export C100_RUNS=8 C100_EPOCHS=5 C100_TARGET=0.70 C100_SLEEP_CYCLES=0 C100_BATCH=1024 C100_COMPILE=1
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

run_arm s3p_5
run_arm wide96_5   C100_WIDTHS=96,192,384
run_arm wide80_5   C100_WIDTHS=80,160,320
run_arm hotlr_5    C100_MUON_LR=0.065
# capacity-vs-time reference: moderate-wide @6ep (cheaper than 96,192,384)
run_arm wide80_6   C100_EPOCHS=6 C100_WIDTHS=80,160,320

echo "==> 5ep probe done $(date)"

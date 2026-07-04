#!/bin/bash
# Fine-tune the progressive-resize resolution around the E7 record (20px, 8.806s,
# 0.7105, +1.05pp mean margin). 16px was too aggressive (0.7016); try 18/19/22px and
# a larger low-res fraction to see if a cleaner-or-faster point exists. All with the
# dynamo cache fix so timing stays stable. 8-seed paired screen @7ep, sleep 0.
#SBATCH --job-name=c100-rsztune-dcrisost
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
#SBATCH --output=./logs/rsztune-%j.out
#SBATCH --error=./logs/rsztune-%j.err
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
export C100_RESNET_D=1 C100_LOOKAHEAD_K=5 C100_BN_SHIFT_LR_MULT=8 C100_RESIZE_FRAC=0.5
export C100_DYNAMO_CACHE_LIMIT=256

run_arm() {
  name="$1"; shift
  echo ">>> ARM ${name}"
  env "$@" python train_cifar100_resnet_muon_exp.py || echo ">>> ARM ${name} FAILED rc=$?"
}

run_arm resize20_ref  C100_RESIZE_PX=20
run_arm resize18      C100_RESIZE_PX=18
run_arm resize19      C100_RESIZE_PX=19
run_arm resize18_f60  C100_RESIZE_PX=18 C100_RESIZE_FRAC=0.6
run_arm resize20_f60  C100_RESIZE_PX=20 C100_RESIZE_FRAC=0.6

echo "==> resize tune done $(date)"

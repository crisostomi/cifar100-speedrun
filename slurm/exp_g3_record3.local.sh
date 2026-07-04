#!/bin/bash
# G3 record-UPDATE retry with the dynamo cache-stability fix. E6 showed resize20 is a
# clean 8.87s on 28/30 runs but 2 runs recompiled inside the timer (dynamo cache
# eviction). C100_DYNAMO_CACHE_LIMIT=256 (infra-only, semantics-preserved) should stop
# the eviction/recompile. Paired same-node: resize24 (current E2 record) vs resize20,
# 30 official runs, sleep 1e9, sb880000. If resize20 firms to ~8.87s std<0.05 and 30/30
# -> new record superseding E2.
#SBATCH --job-name=c100-g3rec3-dcrisost
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
#SBATCH --output=./logs/g3rec3-%j.out
#SBATCH --error=./logs/g3rec3-%j.err
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

export C100_RUNS=30 C100_EPOCHS=7 C100_TARGET=0.70 C100_SLEEP_CYCLES=1000000000 C100_SEED_BASE=880000 C100_BATCH=1024 C100_COMPILE=1
export C100_PRECISION=bf16 C100_LABEL_SMOOTHING=0.15 C100_SCHEDULE=wsd
export C100_WARMUP_FRAC=0.05 C100_COOLDOWN_FRAC=0.40
export C100_MUON_NESTEROV=1 C100_NS_STEPS=3 C100_MUON_MOM_WARMUP=0.15 C100_MUON_MOM_START=0.85
export C100_CROP_PAD=3 C100_MUON_LR=0.055
export C100_RESNET_D=1 C100_LOOKAHEAD_K=5 C100_BN_SHIFT_LR_MULT=8 C100_RESIZE_FRAC=0.5
# Infra-only compile-stability fix, applied to BOTH arms for a fair paired comparison.
export C100_DYNAMO_CACHE_LIMIT=256

run_arm() {
  name="$1"; shift
  echo ">>> ARM ${name}"
  env "$@" python train_cifar100_resnet_muon_exp.py || echo ">>> ARM ${name} FAILED rc=$?"
}

run_arm s3p_7_rec24   C100_RESIZE_PX=24
run_arm resize20_7    C100_RESIZE_PX=20

echo "==> g3-record3 done $(date)"

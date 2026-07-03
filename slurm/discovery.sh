#!/bin/bash
#SBATCH --job-name=c100-discovery
#SBATCH --account=IscrC_SIMP
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/leonardo_work/IscrC_YENDRI/paerle/Cifar100Speedrun/logs/discovery-%j.out
#SBATCH --error=/leonardo_work/IscrC_YENDRI/paerle/Cifar100Speedrun/logs/discovery-%j.err
set -euo pipefail
if [[ "${SLURM_JOB_ACCOUNT:-}" != "iscrc_simp" && "${SLURM_JOB_ACCOUNT:-}" != "IscrC_SIMP" ]]; then
  echo "Refusing to run outside IscrC_SIMP." >&2
  exit 2
fi
cd /leonardo_work/IscrC_YENDRI/paerle/Cifar100Speedrun
source env_setup.sh
echo "==> $(date) job=${SLURM_JOB_ID:-N/A} node=$(hostname)"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
python prepare_cifar100_hf.py
RUNS=${RUNS:-5}
TARGET=${TARGET:-0.50}
for EPOCHS in ${EPOCHS_LIST:-4 8 12 16 20}; do
  echo "===== CIFAR100 discovery epochs=${EPOCHS} runs=${RUNS} target=${TARGET} ====="
  C100_RUNS=$RUNS C100_EPOCHS=$EPOCHS C100_SLEEP_CYCLES=1000000000 python train_cifar100_resnet_muon.py
  echo
 done
echo "==> done $(date)"

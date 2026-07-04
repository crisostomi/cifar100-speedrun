#!/bin/bash
# Tranche-1 STACK + 6-epoch floor-break screen. Combines the promoted margin levers
# (ResNet-D + Lookahead + BN-beta LR) and tests whether the gained margin lets the
# run drop to 6 epochs (Raimon's declared hard floor) while holding mean val > 70%.
# Also stacks the HOLD speed lever (progressive resize) for the fast frontier.
# One job = same-node paired vs champion refs. 8 seeds. Screening, not a record run.
#SBATCH --job-name=c100-tr1stk-dcrisost
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
#SBATCH --output=./logs/tr1stk-%j.out
#SBATCH --error=./logs/tr1stk-%j.err
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

run_arm() {
  name="$1"; shift
  echo ">>> ARM ${name}"
  env "$@" python train_cifar100_resnet_muon_exp.py || echo ">>> ARM ${name} FAILED rc=$?"
}

# References
run_arm champ7
run_arm champ6    C100_EPOCHS=6
# Stack of promoted margin levers (ResNet-D + Lookahead + BN-beta) at 7 and 6 ep
run_arm s3_7      C100_RESNET_D=1 C100_LOOKAHEAD_K=5 C100_BN_SHIFT_LR_MULT=8
run_arm s3_6      C100_EPOCHS=6 C100_RESNET_D=1 C100_LOOKAHEAD_K=5 C100_BN_SHIFT_LR_MULT=8
# Two-lever variant (drop the marginal BN-beta) at 6 ep
run_arm s2_6      C100_EPOCHS=6 C100_RESNET_D=1 C100_LOOKAHEAD_K=5
# Stack + progressive resize (fast frontier) at 7 and 6 ep
run_arm s3p_7     C100_RESNET_D=1 C100_LOOKAHEAD_K=5 C100_BN_SHIFT_LR_MULT=8 C100_RESIZE_PX=24 C100_RESIZE_FRAC=0.5
run_arm s3p_6     C100_EPOCHS=6 C100_RESNET_D=1 C100_LOOKAHEAD_K=5 C100_BN_SHIFT_LR_MULT=8 C100_RESIZE_PX=24 C100_RESIZE_FRAC=0.5

echo "==> tranche1-stack done $(date)"

#!/bin/bash
# SLURM job: fine-tune PET (VGG16-bn) on wheat tips or pea pods, from the crowd
# (ShanghaiTech-A) checkpoint.
#
# Prerequisites:
#   - pet_final/pretrained/pet_sha_model_only.pth  (crowd checkpoint — via strip_ckpt.py)
#   - PET/pretrained/vgg16_bn-6c64b313.pth          (ImageNet backbone)
#   - pet_final/data_<resolution>/ or data_drone_<resolution>/  (via prepare_pet_data.sh [--drone])
#
# Usage (from the wpcount/ root):
#   sbatch pet_final/train.sh --wheat
#   sbatch pet_final/train.sh --pea
#   sbatch pet_final/train.sh --wheat --resolution 1500
#   sbatch pet_final/train.sh --wheat --drone --resolution 2048
#
#SBATCH --job-name=train_pet
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:ampere:1
#SBATCH --time=4-00:00:00
#SBATCH --partition=tau

set -euo pipefail

EPOCHS=300
BATCH_SIZE=8
LR=1e-4
LR_BACKBONE=1e-5
WHEAT=false
PEA=false
DRONE=false
RES=2048
ENV_NAME="${PET_ENV:-PET_ENV}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)       EPOCHS="$2";      shift 2 ;;
        --batch_size)   BATCH_SIZE="$2";  shift 2 ;;
        --lr)           LR="$2";          shift 2 ;;
        --lr_backbone)  LR_BACKBONE="$2"; shift 2 ;;
        --resume)       OVERRIDE_RESUME="$2"; shift 2 ;;
        --wheat)        WHEAT=true;       shift   ;;
        --pea)          PEA=true;         shift   ;;
        --drone)        DRONE=true;       shift   ;;
        --resolution)   RES="$2";         shift 2 ;;
        --env)          ENV_NAME="$2";    shift 2 ;;
        *)              shift ;;
    esac
done

if [[ "$WHEAT" == true && "$PEA" == true ]]; then
    echo "ERROR: --wheat and --pea are mutually exclusive."; exit 1
fi
if [[ "$WHEAT" != true && "$PEA" != true ]]; then
    echo "ERROR: select a species with --wheat or --pea."; exit 1
fi

SPECIES_DIR="pea" && [[ "$PEA" != true ]] && SPECIES_DIR="wheat"
DRONE_TAG="" && [[ "$DRONE" == true ]] && DRONE_TAG="drone_"
DATA_SUBDIR="data_${DRONE_TAG}${RES}"
OUTPUT_DIR="pet_${DRONE_TAG}${SPECIES_DIR}_${RES}"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT"
mkdir -p pet_final_out
_TS=$(date +%Y%m%d_%H%M%S)
exec > "pet_final_out/train_pet_${DRONE_TAG}${SPECIES_DIR}_${RES}_${_TS}.out" 2>&1

RESUME="$ROOT/pet_final/pretrained/pet_sha_model_only.pth"
[[ -n "${OVERRIDE_RESUME:-}" ]] && RESUME="$OVERRIDE_RESUME"

echo "============================================================"
echo "Task     : fine-tune PET (${SPECIES_DIR}${DRONE_TAG:+, drone})   resolution=${RES} (${DATA_SUBDIR})"
echo "Job      : ${SLURM_JOB_NAME:-local} (ID ${SLURM_JOB_ID:-local})"
echo "Node     : $(hostname)   GPU : ${CUDA_VISIBLE_DEVICES:-n/a}"
echo "Epochs   : ${EPOCHS}   Batch : ${BATCH_SIZE}   LR : ${LR}   LR_bb : ${LR_BACKBONE}   Env : ${ENV_NAME}"
echo "Resume   : $RESUME"
echo "Started  : $(date)   Timecode : ${_TS}"
echo "============================================================"

# Per-job work dir: each job gets its own data/ShanghaiTech/part_A so concurrent PET
# jobs don't overwrite each other's symlink (PET resolves part_A relative to CWD).
JOB_PET="$ROOT/PET_jobs/${SLURM_JOB_ID:-$$}"
mkdir -p "$JOB_PET/data/ShanghaiTech"
for item in "$ROOT/PET"/*/; do
    name=$(basename "${item%/}")
    [[ "$name" == "data" || "$name" == "outputs" ]] && continue
    ln -sfn "${item%/}" "$JOB_PET/$name"
done
for item in "$ROOT/PET"/*; do
    [[ -f "$item" ]] && ln -sfn "$item" "$JOB_PET/$(basename "$item")"
done
ln -sfn "$ROOT/PET/outputs" "$JOB_PET/outputs"
ln -sfn "$ROOT/pet_final/${DATA_SUBDIR}/${SPECIES_DIR}" "$JOB_PET/data/ShanghaiTech/part_A"
echo "PET job dir : $JOB_PET"
echo "Symlink : data/ShanghaiTech/part_A -> pet_final/${DATA_SUBDIR}/${SPECIES_DIR}"

cd "$JOB_PET"
echo "Python  : $(which python)"
echo "PyTorch : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA    : $(python -c 'import torch; print(torch.version.cuda)')"
echo "------------------------------------------------------------"

export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((10000 + ${SLURM_JOB_ID:-0} % 20000))
echo "Distributed : MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT  (world_size=1)"

SPECIES_ARG="--pea" && [[ "$PEA" != true ]] && SPECIES_ARG="--wheat"

srun python main.py \
    --dataset_file SHA \
    --resume       "$RESUME" \
    --output_dir   "$OUTPUT_DIR" \
    --epochs       "$EPOCHS" \
    --batch_size   "$BATCH_SIZE" \
    --lr           "$LR" \
    --lr_backbone  "$LR_BACKBONE" \
    --eval_freq    5 \
    --resolution   "$RES" \
    $SPECIES_ARG

echo "============================================================"
echo "Done : $(date)"
echo "Checkpoints : PET/outputs/SHA/${OUTPUT_DIR}/{checkpoint.pth,best_checkpoint.pth}"
echo "============================================================"
echo "=== PET fine-tuning finished ==="

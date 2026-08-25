#!/bin/bash
# SLURM job: pure PET inference on a folder of images. Species and resolution
# are read from the checkpoint path (pet_<wheat|pea>_<resolution> convention).
#
# Usage:
#   sbatch pet_final/infer.sh --images /path/to/images --ckpt PET/outputs/SHA/pet_wheat_2048/best_checkpoint.pth
#   sbatch pet_final/infer.sh --images /path/to/images --ckpt PET/outputs/SHA/pet_pea_1500/best_checkpoint.pth
#
#SBATCH --job-name=infer_pet
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

IMAGES=""
CKPT=""
ENV_NAME="${PET_ENV:-PET_ENV}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --images) IMAGES="$2"; shift 2 ;;
        --ckpt)   CKPT="$2";   shift 2 ;;
        --env)    ENV_NAME="$2"; shift 2 ;;
        *) shift ;;
    esac
done

if [[ -z "$IMAGES" || -z "$CKPT" ]]; then
    echo "ERROR: --images and --ckpt are required."; exit 1
fi
if [[ ! -d "$IMAGES" ]]; then
    echo "ERROR: images folder not found: $IMAGES"; exit 1
fi
if [[ ! -f "$CKPT" ]]; then
    echo "ERROR: checkpoint not found: $CKPT"; exit 1
fi

source ~/miniforge3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT"
mkdir -p pet_final_out

# Best-effort parse of the checkpoint path for the log filename only;
# infer.py does the authoritative parsing and errors out if it can't.
TAG="unknown"
[[ "$CKPT" =~ pet_(drone_)?(wheat|pea)_([0-9]+) ]] && TAG="${BASH_REMATCH[1]}${BASH_REMATCH[2]}_${BASH_REMATCH[3]}"

_TS=$(date +%Y%m%d_%H%M%S)
exec > "pet_final_out/infer_pet_${TAG}_${_TS}.out" 2>&1

echo "============================================================"
echo "Task    : pure PET inference  (${TAG})"
echo "Images  : $IMAGES"
echo "Model   : $CKPT"
echo "Node    : $(hostname)   GPU : ${CUDA_VISIBLE_DEVICES:-n/a}   Env : ${ENV_NAME}"
echo "Started : $(date)"
echo "============================================================"

JOB_PET="$ROOT/PET_jobs/${SLURM_JOB_ID:-$$}_infer"
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

# dummy symlink so PET's dataset path doesn't fail at import
ln -sfn "$ROOT/pet_final/data_2048/wheat_test" \
    "$JOB_PET/data/ShanghaiTech/part_A" 2>/dev/null || true

cd "$JOB_PET"
srun python "$ROOT/pet_final/infer.py" \
    --images "$IMAGES" \
    --ckpt   "$CKPT"

echo "============================================================"
echo "Done : $(date)   Output -> pet_final/inference/"
echo "============================================================"
echo "=== PET inference finished ==="

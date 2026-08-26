#!/bin/bash
# SLURM job: data-size learning curve. Trains the same species/resolution PET
# model at several increasing training-set sizes — fixed-seed nested subsets
# of the SAME already-prepared pet_final/data_<resolution>/<species>/train_data/,
# no new data directory is created — then plots best val MAE vs number of
# training images used.
#
# Prerequisites: same as train.sh (crowd checkpoint, VGG backbone, prepared data).
#
# Usage (from the wpcount/ root):
#   sbatch pet_final/size_curve.sh --wheat
#   sbatch pet_final/size_curve.sh --pea --resolution 1500
#   sbatch pet_final/size_curve.sh --wheat --sizes 20 40 60 80 100
#   sbatch pet_final/size_curve.sh --wheat --drone --epochs 150
#
#SBATCH --job-name=size_curve_pet
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

EPOCHS=150
BATCH_SIZE=8
LR=1e-4
LR_BACKBONE=1e-5
WHEAT=false
PEA=false
DRONE=false
RES=2048
SIZES=()
ENV_NAME="${PET_ENV:-PET_ENV}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --epochs)       EPOCHS="$2";      shift 2 ;;
        --batch_size)   BATCH_SIZE="$2";  shift 2 ;;
        --lr)           LR="$2";          shift 2 ;;
        --lr_backbone)  LR_BACKBONE="$2"; shift 2 ;;
        --wheat)        WHEAT=true;       shift   ;;
        --pea)          PEA=true;         shift   ;;
        --drone)        DRONE=true;       shift   ;;
        --resolution)   RES="$2";         shift 2 ;;
        --env)          ENV_NAME="$2";    shift 2 ;;
        --sizes)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                SIZES+=("$1"); shift
            done
            ;;
        *) shift ;;
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

source ~/miniforge3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT"
mkdir -p pet_final_out

TRAIN_IMG_DIR="pet_final/${DATA_SUBDIR}/${SPECIES_DIR}/train_data/images"
if [[ ! -d "$TRAIN_IMG_DIR" ]]; then
    echo "ERROR: $TRAIN_IMG_DIR not found — run prepare_pet_data.sh first."; exit 1
fi
N_TOTAL=$(ls "$TRAIN_IMG_DIR" | wc -l | tr -d ' ')

# Default sizes: 20/40/60/80/100% of the available training images, deduplicated.
if [[ ${#SIZES[@]} -eq 0 ]]; then
    SIZES=($(python3 -c "
n = $N_TOTAL
fracs = [0.2, 0.4, 0.6, 0.8, 1.0]
sizes = sorted(set(max(1, round(n * f)) for f in fracs))
print(' '.join(str(s) for s in sizes))
"))
fi

_TS=$(date +%Y%m%d_%H%M%S)
exec > "pet_final_out/size_curve_pet_${DRONE_TAG}${SPECIES_DIR}_${RES}_${_TS}.out" 2>&1

echo "============================================================"
echo "Task        : data-size learning curve (${SPECIES_DIR}${DRONE_TAG:+, drone})   resolution=${RES}"
echo "Sizes       : ${SIZES[*]}   (out of ${N_TOTAL} available training images)"
echo "Job         : ${SLURM_JOB_NAME:-local} (ID ${SLURM_JOB_ID:-local})"
echo "Node        : $(hostname)   GPU : ${CUDA_VISIBLE_DEVICES:-n/a}"
echo "Epochs/size : ${EPOCHS}   Batch : ${BATCH_SIZE}   LR : ${LR}   LR_bb : ${LR_BACKBONE}   Env : ${ENV_NAME}"
echo "Started     : $(date)   Timecode : ${_TS}"
echo "============================================================"

RESUME="$ROOT/pet_final/pretrained/pet_sha_model_only.pth"

# Per-job work dir, reused across all sizes in this sweep (same species/resolution/
# drone-ness throughout — only --train_subset and --output_dir change per iteration).
JOB_PET="$ROOT/PET_jobs/${SLURM_JOB_ID:-$$}_size_curve"
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

cd "$JOB_PET"
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((10000 + ${SLURM_JOB_ID:-0} % 20000))

SPECIES_ARG="--pea" && [[ "$PEA" != true ]] && SPECIES_ARG="--wheat"
DRONE_ARG="" && [[ "$DRONE" == true ]] && DRONE_ARG="--drone"

for N in "${SIZES[@]}"; do
    OUTPUT_DIR="size_curve_${DRONE_TAG}${SPECIES_DIR}_${RES}_n${N}"

    # A batch can't be larger than the subset itself.
    THIS_BATCH=$BATCH_SIZE
    if (( N < THIS_BATCH )); then THIS_BATCH=$N; fi

    echo "------------------------------------------------------------"
    echo "Training on ${N}/${N_TOTAL} images (batch_size=${THIS_BATCH}) -> ${OUTPUT_DIR}"
    echo "------------------------------------------------------------"
    srun python main.py \
        --dataset_file SHA \
        --resume       "$RESUME" \
        --output_dir   "$OUTPUT_DIR" \
        --epochs       "$EPOCHS" \
        --batch_size   "$THIS_BATCH" \
        --lr           "$LR" \
        --lr_backbone  "$LR_BACKBONE" \
        --eval_freq    5 \
        --resolution   "$RES" \
        --train_subset "$N" \
        $SPECIES_ARG \
        $DRONE_ARG
done

cd "$ROOT"
echo "============================================================"
echo "All sizes done — aggregating"
echo "============================================================"
python pet_final/size_curve_aggregate.py \
    $SPECIES_ARG \
    --resolution "$RES" \
    $DRONE_ARG \
    --sizes "${SIZES[@]}" \
    --timestamp "$_TS"

echo "============================================================"
echo "Done : $(date)"
echo "============================================================"
echo "=== PET data-size learning curve finished ==="

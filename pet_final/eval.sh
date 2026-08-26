#!/bin/bash
# SLURM job: evaluate a fine-tuned PET checkpoint on the test split (LOCA-style metrics).
#
# Usage (from the wpcount/ root):
#   sbatch pet_final/eval.sh --wheat
#   sbatch pet_final/eval.sh --pea --resolution 1500
#   sbatch pet_final/eval.sh --wheat --bordure
#   sbatch pet_final/eval.sh --wheat --px_wheat 25
#   sbatch pet_final/eval.sh --pea   --px_pea 50
#   sbatch pet_final/eval.sh --wheat --drone --resolution 2048
#   sbatch pet_final/eval.sh --wheat --resume PET/outputs/SHA/pet_wheat_1500/best_checkpoint.pth
#
#SBATCH --job-name=eval_pet
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

RESUME=""
WHEAT=false
PEA=false
DRONE=false
BORDURE=""
PX_WHEAT=""
PX_PEA=""
RES=2048
ENV_NAME="${PET_ENV:-PET_ENV}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)     RESUME="$2";     shift 2 ;;
        --wheat)      WHEAT=true;      shift   ;;
        --pea)        PEA=true;        shift   ;;
        --drone)      DRONE=true;      shift   ;;
        --bordure)
            if [[ -n "${2:-}" && "${2:-}" =~ ^[0-9]+$ ]]; then
                BORDURE="$2"; shift 2
            else
                BORDURE=20; shift
            fi
            ;;
        --px_wheat)   PX_WHEAT="$2";  shift 2 ;;
        --px_pea)     PX_PEA="$2";    shift 2 ;;
        --resolution) RES="$2";        shift 2 ;;
        --env)        ENV_NAME="$2";   shift 2 ;;
        *)            shift ;;
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
BORDURE_SUFFIX="" && [[ -n "$BORDURE" ]] && BORDURE_SUFFIX="_bordure"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$ROOT"
mkdir -p pet_final_out
_TS=$(date +%Y%m%d_%H%M%S)
exec > "pet_final_out/eval_pet_${DRONE_TAG}${SPECIES_DIR}_${RES}${BORDURE_SUFFIX}_${_TS}.out" 2>&1

CKPT_ROOT="$ROOT/PET/outputs"
[[ -z "$RESUME" ]] && RESUME="${CKPT_ROOT}/SHA/${OUTPUT_DIR}/best_checkpoint.pth"
if [[ ! -f "$RESUME" ]]; then
    echo "ERROR: checkpoint not found: $RESUME"; exit 1
fi

echo "============================================================"
echo "Task    : eval PET (${SPECIES_DIR}${DRONE_TAG:+, drone}, test split)   resolution=${RES} (${DATA_SUBDIR})"
echo "Model   : $RESUME"
echo "Node    : $(hostname)   GPU : ${CUDA_VISIBLE_DEVICES:-n/a}   Env : ${ENV_NAME}"
echo "Started : $(date)"
echo "============================================================"

# Per-job work dir: each job gets its own data/ShanghaiTech/part_A — no symlink race.
JOB_PET="$ROOT/PET_jobs/${SLURM_JOB_ID:-$$}_eval"
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
ln -sfn "$ROOT/pet_final/${DATA_SUBDIR}/${SPECIES_DIR}_test" "$JOB_PET/data/ShanghaiTech/part_A"
echo "PET job dir : $JOB_PET"
echo "Symlink : data/ShanghaiTech/part_A -> pet_final/${DATA_SUBDIR}/${SPECIES_DIR}_test"

SPECIES_ARG="--pea" && [[ "$PEA" != true ]] && SPECIES_ARG=""
DRONE_ARG=""   && [[ "$DRONE" == true ]] && DRONE_ARG="--drone"
BORDURE_ARG=""   && [[ -n "$BORDURE"  ]] && BORDURE_ARG="--bordure $BORDURE"
PX_WHEAT_ARG=""  && [[ -n "$PX_WHEAT" ]] && PX_WHEAT_ARG="--px_wheat $PX_WHEAT"
PX_PEA_ARG=""    && [[ -n "$PX_PEA"   ]] && PX_PEA_ARG="--px_pea $PX_PEA"

cd "$JOB_PET"
srun python "$ROOT/pet_final/eval_metrics.py" \
    --dataset_file  SHA \
    --resume        "$RESUME" \
    --resolution    "$RES" \
    $SPECIES_ARG $DRONE_ARG $BORDURE_ARG $PX_WHEAT_ARG $PX_PEA_ARG

echo "============================================================"
echo "Done : $(date)   Outputs : pet_final/outputs_${DRONE_TAG}${SPECIES_DIR}_${RES}${BORDURE_SUFFIX}/"
echo "============================================================"
echo "=== PET eval finished ==="

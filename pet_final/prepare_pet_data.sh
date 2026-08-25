#!/bin/bash
# SLURM job: convert wheat + pea point annotations into PET's ShanghaiTech layout.
#
# Usage (from the wpcount/ root):
#   sbatch pet_final/prepare_pet_data.sh
#   sbatch pet_final/prepare_pet_data.sh --resolution 1500
#   sbatch pet_final/prepare_pet_data.sh --resolution 1500 2500 3000
#   sbatch pet_final/prepare_pet_data.sh --path /custom/annotations --resolution 2048
#   sbatch pet_final/prepare_pet_data.sh --drone --resolution 2048
#
#SBATCH --job-name=prep_pet
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=4-00:00:00
#SBATCH --partition=normal-best

set -euo pipefail

PATH_ARG=""
DRONE=false
RES_LIST=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --path)
            PATH_ARG="$2"; shift 2 ;;
        --drone)
            DRONE=true; shift ;;
        --resolution)
            shift
            while [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; do
                RES_LIST+=("$1"); shift
            done
            ;;
        *) shift ;;
    esac
done

DRONE_TAG="" && [[ "$DRONE" == true ]] && DRONE_TAG="drone_"

source ~/miniforge3/etc/profile.d/conda.sh
conda activate PET_ENV

cd "$SLURM_SUBMIT_DIR"
mkdir -p pet_final_out
_TS=$(date +%Y%m%d_%H%M%S)
_RES_TAG="2048"
[[ ${#RES_LIST[@]} -gt 0 ]] && _RES_TAG=$(IFS=_; echo "${RES_LIST[*]}")
exec > "pet_final_out/prep_pet_${DRONE_TAG}${_RES_TAG}_${_TS}.out" 2>&1

echo "============================================================"
echo "Task    : prepare_pet_data  drone=${DRONE}  path=${PATH_ARG:-<default>}  resolution=${RES_LIST[*]:-2048}"
echo "Job     : ${SLURM_JOB_NAME:-local}  (ID ${SLURM_JOB_ID:-local})"
echo "Node    : $(hostname)"
echo "Started : $(date)   Timecode : ${_TS}"
echo "============================================================"

CMD=(python pet_final/prepare_pet_data.py)
[[ -n "$PATH_ARG" ]] && CMD+=(--path "$PATH_ARG")
[[ "$DRONE" == true ]] && CMD+=(--drone)
[[ ${#RES_LIST[@]} -gt 0 ]] && CMD+=(--resolution "${RES_LIST[@]}")
srun "${CMD[@]}"

echo "============================================================"
echo "Done : $(date)"
echo "============================================================"
echo "=== PET data preparation finished ==="

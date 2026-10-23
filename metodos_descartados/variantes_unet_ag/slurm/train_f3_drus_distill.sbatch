#!/bin/bash
#SBATCH --job-name=v2_f3drus
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/v2_f3drus_%j.out
#SBATCH --error=slurm_logs/v2_f3drus_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<tu_correo>
#
# Fase F.3 — destilación DRUS-noFT: reentrena unet_ag (misma config anchor05
# que el baseline: anchor_w=0.5, phantom_type=both, pulse aug, 20 ép) pero
# con los targets del teacher DDRM en vez de los Schiffner-points.
#
# Los targets vienen de:
#   1) teacher DDRM  (sample_ddrm --phantoms all → fieldii_teacher_noft/x_*.npy)
#   2) make_drus_targets.py → unet_ag/targets_drus_noft/<ptype>/<id>/compounded.npy
# Se consumen con --target_is_bmode: la salida DRUS YA es B-mode log-comprimida
# en [-1,1]; el trainer sólo la remapea a [0,1] y NO re-aplica hilbert/log
# (re-envolverla la destruiría).
#
# PREREQUISITO (una vez, tras terminar el teacher 206682):
#   source env.sh    # PYTHONPATH — imprescindible o ModuleNotFoundError unet_ag
#   python zhang_drus/scripts/make_drus_targets.py \
#       --teacher_dir zhang_drus/results_drus/fieldii_teacher_noft \
#       --out unet_ag/targets_drus_noft --gamma 25 --seed 0
#
# Smoke (validar el pipeline de target antes del full):
#   source env.sh
#   sbatch --export=ALL,EPOCHS=2,MAX_PHANTOMS_TRAIN=6,MAX_PHANTOMS_VAL=4,SMOKE_STEPS=6 \
#          unet_ag/unet_ag/slurm/train_f3_drus_distill.sbatch
#
# Full:
#   source env.sh
#   sbatch unet_ag/unet_ag/slurm/train_f3_drus_distill.sbatch

set -euo pipefail
mkdir -p slurm_logs

DATA_ROOT="${DATA_ROOT:-${SLURM_SUBMIT_DIR}/datasets/Field II/field_ii_data_multiangle}"
TARGET_ROOT="${TARGET_ROOT:-${SLURM_SUBMIT_DIR}/unet_ag/targets_drus_noft}"
OUTPUT_DIR="${OUTPUT_DIR:-${SLURM_SUBMIT_DIR}/unet_ag/results/b1_v5_anchor05_drustgt}"
PHANTOM_TYPE="${PHANTOM_TYPE:-both}"
EPOCHS="${EPOCHS:-20}"
LR="${LR:-1e-3}"
WD="${WD:-1e-4}"
NZ="${NZ:-612}"
NX="${NX:-388}"
BASE_CH="${BASE_CH:-32}"
EMBED_CH="${EMBED_CH:-16}"
MSE_WEIGHT="${MSE_WEIGHT:-0.7}"
L1_WEIGHT="${L1_WEIGHT:-0.2}"
SSIM_WEIGHT="${SSIM_WEIGHT:-0.1}"
ANCHOR_WEIGHT="${ANCHOR_WEIGHT:-0.5}"      # anchor05 — igual que el baseline
GAIN_REG_WEIGHT="${GAIN_REG_WEIGHT:-0.0}"
INIT_GAIN="${INIT_GAIN:-1.0}"
SEED="${SEED:-42}"
MAX_PHANTOMS_TRAIN="${MAX_PHANTOMS_TRAIN:-}"
MAX_PHANTOMS_VAL="${MAX_PHANTOMS_VAL:-}"
SMOKE_STEPS="${SMOKE_STEPS:-}"
RESUME="${RESUME:-0}"
NO_AUG="${NO_AUG:-0}"
FREEZE_GAIN="${FREEZE_GAIN:-0}"
PULSE_APPLY_PROB="${PULSE_APPLY_PROB:-0.7}"
PULSE_BW_MIN_HZ="${PULSE_BW_MIN_HZ:-1.0e6}"
PULSE_BW_MAX_HZ="${PULSE_BW_MAX_HZ:-2.5e6}"
PULSE_FC_SHIFT_HZ="${PULSE_FC_SHIFT_HZ:-300e3}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate tesis_us

cd "$SLURM_SUBMIT_DIR"

echo "=== Job ${SLURM_JOB_ID} F.3 DRUS-distill on $(hostname) — $(date) ==="
echo "  data_root       : $DATA_ROOT"
echo "  target_root     : $TARGET_ROOT  (DRUS-noFT teacher, --target_is_bmode)"
echo "  output_dir      : $OUTPUT_DIR"
echo "  phantom_type    : $PHANTOM_TYPE"
echo "  epochs          : $EPOCHS  lr : $LR  wd : $WD"
echo "  grid            : ${NZ}x${NX}"
echo "  loss weights    : mse=$MSE_WEIGHT  l1=$L1_WEIGHT  ssim=$SSIM_WEIGHT"
echo "  anchor_w        : $ANCHOR_WEIGHT  gain_reg_w : $GAIN_REG_WEIGHT"
echo "  pulse aug       : prob=$PULSE_APPLY_PROB  bw=[${PULSE_BW_MIN_HZ},${PULSE_BW_MAX_HZ}]  fc_shift=±${PULSE_FC_SHIFT_HZ}"
[ -n "$MAX_PHANTOMS_TRAIN" ] && echo "  max_train       : $MAX_PHANTOMS_TRAIN"
[ -n "$MAX_PHANTOMS_VAL"   ] && echo "  max_val         : $MAX_PHANTOMS_VAL"
[ -n "$SMOKE_STEPS"        ] && echo "  smoke_steps     : $SMOKE_STEPS"

if [ ! -d "$TARGET_ROOT/points" ] && [ ! -d "$TARGET_ROOT/cysts" ]; then
    echo "!! ERROR: $TARGET_ROOT vacío — corre make_drus_targets.py primero." >&2
    exit 1
fi
echo "  n targets       : points=$(ls -d $TARGET_ROOT/points/*/ 2>/dev/null | wc -l)  cysts=$(ls -d $TARGET_ROOT/cysts/*/ 2>/dev/null | wc -l)"
echo "============================================"

EXTRA_ARGS=()
[ -n "$MAX_PHANTOMS_TRAIN" ] && EXTRA_ARGS+=(--max_phantoms_train "$MAX_PHANTOMS_TRAIN")
[ -n "$MAX_PHANTOMS_VAL"   ] && EXTRA_ARGS+=(--max_phantoms_val   "$MAX_PHANTOMS_VAL")
[ -n "$SMOKE_STEPS"        ] && EXTRA_ARGS+=(--smoke_steps        "$SMOKE_STEPS")
[ "$RESUME" = "1" ]          && EXTRA_ARGS+=(--resume)
[ "$NO_AUG" = "1" ]          && EXTRA_ARGS+=(--no_aug)
[ "$FREEZE_GAIN" = "1" ]     && EXTRA_ARGS+=(--freeze_gain)

python -u -m unet_ag.train.b1_v5_pulse_aug \
    --data_root         "$DATA_ROOT" \
    --target_root       "$TARGET_ROOT" \
    --output_dir        "$OUTPUT_DIR" \
    --phantom_type      "$PHANTOM_TYPE" \
    --target_is_bmode \
    --epochs            "$EPOCHS" --lr "$LR" --weight_decay "$WD" \
    --nz "$NZ" --nx "$NX" \
    --base_ch           "$BASE_CH" --embed_ch "$EMBED_CH" \
    --mse_weight        "$MSE_WEIGHT" \
    --l1_weight         "$L1_WEIGHT" \
    --ssim_weight       "$SSIM_WEIGHT" \
    --anchor_weight     "$ANCHOR_WEIGHT" \
    --gain_reg_weight   "$GAIN_REG_WEIGHT" \
    --init_gain         "$INIT_GAIN" \
    --pulse_apply_prob  "$PULSE_APPLY_PROB" \
    --pulse_bw_min_hz   "$PULSE_BW_MIN_HZ" \
    --pulse_bw_max_hz   "$PULSE_BW_MAX_HZ" \
    --pulse_fc_shift_hz "$PULSE_FC_SHIFT_HZ" \
    --seed              "$SEED" \
    --device cuda \
    "${EXTRA_ARGS[@]}"

echo "=== Done — $(date) ==="

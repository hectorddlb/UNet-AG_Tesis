#!/bin/bash
#SBATCH --job-name=v2_prefilter
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/v2_prefilter_%j.out
#SBATCH --error=slurm_logs/v2_prefilter_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<tu_correo>
#
# (B) — Diagnóstico OOD penalty: grid search 5×5 sobre (fc, BW) Gaussian
# bandpass aplicado al RF PICMUS antes del adjoint en B1-v4(0.3).
#
# Medición empírica (2026-05-29) del gap espectral Field II ↔ PICMUS:
#   Field II training:        fc_peak=4.97 MHz   BW(-6dB)=1.82 MHz
#   PICMUS exp resolution:    fc_peak=5.04 MHz   BW(-6dB)=1.04 MHz
#
# Si algún (fc, BW) ≠ identity da val_total < identity en phantoms exp
# con margen ≥3% → gap espectral confirmado → defendible como ablation y
# motivación de (A) augment pulse-shape durante train completo (B1-v5).
# Si converge al mismo val_total en todo el grid → necesita intervención
# durante train completo. (A) sigue siendo la vía obligatoria.
#
# Coste estimado: 104 evals (4 phantoms × 26 pares (fc,BW,id)) ≈ 10-15 min.

set -euo pipefail

mkdir -p slurm_logs

CKPT="${CKPT:-${SLURM_SUBMIT_DIR}/results_v2/b1_v4_a0p3/checkpoints/best.pt}"
PICMUS_ROOT="${PICMUS_ROOT:-${SLURM_SUBMIT_DIR}/datasets/PICMUS}"
OUTPUT_DIR="${OUTPUT_DIR:-${SLURM_SUBMIT_DIR}/results_v2/b1_v4_a0p3_prefilter_grid}"
NZ="${NZ:-612}"
NX="${NX:-388}"
BASE_CH="${BASE_CH:-32}"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate tesis_us

cd "$SLURM_SUBMIT_DIR"

echo "=== Job ${SLURM_JOB_ID} on $(hostname) — $(date) ==="
echo "  ckpt        : $CKPT"
echo "  picmus_root : $PICMUS_ROOT"
echo "  output_dir  : $OUTPUT_DIR"
echo "  grid        : ${NZ}x${NX}"
{ nvidia-smi 2>&1 || true; } > /tmp/nvsmi_${SLURM_JOB_ID:-$$}.txt
head -15 /tmp/nvsmi_${SLURM_JOB_ID:-$$}.txt
rm -f /tmp/nvsmi_${SLURM_JOB_ID:-$$}.txt
echo "============================================"

python -u -m unet_ag.scripts.diagnose_prefilter_picmus \
    --ckpt        "$CKPT" \
    --picmus_root "$PICMUS_ROOT" \
    --output_dir  "$OUTPUT_DIR" \
    --nz          "$NZ" \
    --nx          "$NX" \
    --base_ch     "$BASE_CH" \
    --device      cuda

echo "=== Done — $(date) ==="

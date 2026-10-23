#!/bin/bash
#SBATCH --job-name=res_finales
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000M
#SBATCH --time=08:00:00
#SBATCH --output=slurm_logs/res_finales_%j.out
#SBATCH --error=slurm_logs/res_finales_%j.err
#
# Cadena completa del protocolo de evaluación de la tesis.
# Lanzar desde la raíz del repositorio:
#   TESIS_DATASETS=/ruta/datasets DRUS_ARTIFACTS=/ruta/artefactos \
#       sbatch evaluacion/scripts/run_resultados_finales.sbatch
# Sin artefactos de DRUS, el paso 3 se omite y la evaluación usa los
# evaluacion/npys/drus/ que ya existan.

set -euo pipefail
mkdir -p slurm_logs

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate tesis_us

cd "$SLURM_SUBMIT_DIR"
source env.sh

echo "=== Job ${SLURM_JOB_ID} on $(hostname) — $(date) ==="
nvidia-smi | head -12 || true
echo "============================================"

echo "--- [1/4] Clásicos (DAS y Fd-F, 1 y 75 PW), convención de transmisión corregida"
python evaluacion/scripts/gen_classical_picmus_fixed.py --device cuda

echo "--- [2/4] UNet-AG (B1-v5_anchor05), 3 semillas"
python evaluacion/scripts/gen_b1v5_picmus_seeds.py --device cuda

echo "--- [3/4] DRUS (prior sin ajuste fino, γ de la tesis, semilla 0)"
if python -m drus.cli info >/dev/null 2>&1; then
    python evaluacion/scripts/gen_drus_picmus.py
else
    echo "    artefactos de DRUS no disponibles: se omite (ver metodos/04_drus/README.md)"
fi

echo "--- [4/4] Evaluación unificada + tablas"
python evaluacion/scripts/eval_and_figs_finales.py

echo "=== Done — $(date) ==="

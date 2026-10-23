#!/bin/bash
#SBATCH --job-name=bench_coste
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/bench_coste_%j.out
#SBATCH --error=slurm_logs/bench_coste_%j.err

# Tiempo de pared por imagen de los 6 metodos:
# escenario experimental, das_pw en float32 (precision igualada con el adjunto),
# UNet-AG tal como se ejecuta y DRUS de extremo a extremo, todo en el mismo
# proceso y el mismo nodo. Fijar el nodo con --nodelist para comparar corridas:
# los tiempos reportados se midieron en g-0-10.
#
# --mem=64000M porque DRUS carga Vt.npy (17.2 GB) en RAM: los productos
# espectrales del DDRM corren en CPU (spectral_device="cpu" en ddrm_sampler).
# Lanzar desde la raiz del repositorio con TESIS_DATASETS y DRUS_ARTIFACTS.

set -euo pipefail
mkdir -p slurm_logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate tesis_us
cd "$SLURM_SUBMIT_DIR"
source env.sh

OUT="${OUT:-${SLURM_SUBMIT_DIR}/evaluacion/metricas/tiempo_inferencia_fp32.json}"
PHANTOM="${PHANTOM:-exp_resolution}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

echo "=== bench_coste ${SLURM_JOB_ID} on $(hostname) — $(date) ==="
python -u evaluacion/scripts/tiempo_inferencia.py \
    --device cuda --precision float32 --phantom "$PHANTOM" --out "$OUT"
echo "=== Done — $(date) ==="

#!/bin/bash
#SBATCH --job-name=drus_svd
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=126000M
#SBATCH --time=17:30:00
#SBATCH --output=slurm_logs/drus_svd_%j.out
#SBATCH --error=slurm_logs/drus_svd_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<tu_correo>

set -euo pipefail
mkdir -p slurm_logs

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate tesis_us

cd "$SLURM_SUBMIT_DIR"
source env.sh

# Operador (entrada) y SVD (salida): por omisión, los de $DRUS_ARTIFACTS
# (ver metodos/04_drus/README.md). Lanzar desde la raíz del repositorio.
ART="${DRUS_ARTIFACTS:-metodos/04_drus/artefactos}"
IN_DIR="${IN_DIR:-$ART/operador}"
OUT_DIR="${OUT_DIR:-$ART/svd}"

export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
# libmkl_rt necesita encontrar sus sublibs y libiomp5 (pip mkl las deja en env/lib)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

echo "== drus_svd: IN_DIR=$IN_DIR OUT_DIR=$OUT_DIR =="
echo "== nodo: $(hostname) — $(nproc) cpus =="

# Etapa 1: BH denso (GPU si hay; RAM pico ~4 GB, escribe BH.npy 17 GB)
if [ -f "$OUT_DIR/BH.npy" ]; then
    echo "== BH.npy ya existe, se reutiliza =="
else
    python metodos/04_drus/scripts/run_svd_bh.py --stage bh --in_dir "$IN_DIR" --out_dir "$OUT_DIR"
fi

# Etapa 2: SVD (proceso fresco, sin torch; RAM pico ~112 GiB con gesdd)
python metodos/04_drus/scripts/run_svd_bh.py --stage svd --in_dir "$IN_DIR" --out_dir "$OUT_DIR"

echo "== drus_svd DONE =="

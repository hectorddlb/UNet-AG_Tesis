"""Fase B — BH denso + SVD completa. Dos etapas para aislar memoria:

  stage bh  : BH = B @ H por bloques de columnas (GPU si hay, si no CPU torch),
              escrito directo a BH.npy vía memmap (RAM pico ~4 GB).
  stage svd : carga BH.npy (16 GiB) y corre sgesdd MKL ILP64 (~64 GiB de
              workspace). Fallback automático a la ruta Gram (ssyevd de BHᵀBH,
              ~52 GiB pico) si la RAM disponible no alcanza. Gate G-B1 +
              espectro + U^T·By de los 4 phantoms.

La implementación vive en ``drus.preparacion`` (la misma que usa
``drus prepare svd``); este script conserva la interfaz por etapas con la que se
ejecutó en el clúster (``slurm/svd_bh.sbatch``).

Uso (dentro del sbatch):
    python metodos/04_drus/scripts/run_svd_bh.py --stage bh  --in_dir <operador> --out_dir <svd>
    python metodos/04_drus/scripts/run_svd_bh.py --stage svd --in_dir <operador> --out_dir <svd>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "codigo"))

from drus.preparacion import stage_bh, stage_svd  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["bh", "svd"], required=True)
    ap.add_argument("--in_dir", type=Path, required=True, help="dir fase A (B.npz/H.npz/By_*.npy)")
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "bh":
        stage_bh(args.in_dir, args.out_dir)
    else:
        stage_svd(args.in_dir, args.out_dir)


if __name__ == "__main__":
    main()

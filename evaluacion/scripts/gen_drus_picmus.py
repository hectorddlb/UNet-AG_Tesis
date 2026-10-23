"""DRUS (prior sin ajuste fino) sobre los 4 fantomas de PICMUS, con el γ de la tesis.

Usa el paquete ``drus`` (``drus.inference.DRUS``), con las condiciones de la
evaluación: prior publicado por OpenAI sin ajuste fino, γ por fantoma
(``THESIS_GAMMA``), semilla de muestreo 0, 50 pasos DDRM.

Outputs:
  evaluacion/npys/drus/x_<ph>_g<γ>_s<seed>.npy   salida con signo, 256x256
  evaluacion/npys/drus/grid.npz                  ejes de la rejilla nativa

Run from the repository root, with the DRUS artifacts built
(``drus prepare ...``, see metodos/04_drus/README.md):
  python evaluacion/scripts/gen_drus_picmus.py
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATASETS = Path(os.environ.get("TESIS_DATASETS", ROOT / "datos"))  # PICMUS/ y Field II/, ver datos/README.md
sys.path.insert(0, str(ROOT / "codigo"))

from drus.inference import DRUS, THESIS_GAMMA  # noqa: E402
from drus.observations import PHANTOMS, rf_path  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--picmus", type=Path, default=DATASETS / "PICMUS")
    p.add_argument("--artifacts", type=Path, default=None,
                   help="raíz de artefactos DRUS (por omisión $DRUS_ARTIFACTS)")
    p.add_argument("--out", type=Path, default=ROOT / "evaluacion" / "npys" / "drus")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    modelo = DRUS.from_artifacts(args.artifacts, device=args.device)
    shutil.copyfile(modelo.artifacts.operador / "grid.npz", args.out / "grid.npz")
    t_all = time.perf_counter()
    for ph in PHANTOMS:
        gamma = THESIS_GAMMA[ph]
        for seed in args.seeds:
            res = modelo.reconstruct_picmus(rf_path(args.picmus, ph), phantom=ph,
                                            gamma=gamma, seed=seed)
            out = args.out / f"x_{ph}_g{gamma:g}_s{seed}.npy"
            np.save(out, res.x)
            print(f"  {ph}: γ={gamma:g} seed={seed} {res.seconds:.0f}s -> {out.name}",
                  flush=True)
    print(f"DONE in {(time.perf_counter() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()

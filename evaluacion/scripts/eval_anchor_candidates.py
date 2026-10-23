"""Mide, con el protocolo canonico, las DOS imagenes candidatas a anclaje.

CONTEXTO
--------
El termino de anclaje de la perdida es ``(pred - x_das).abs().mean()`` con
``x_das = inputs["x0"]``, y ``x0`` es ``_envelope_log_compress(OpHFdf.adjoint)``.
Los defaults de OpHFdf son GratingAngleLB(60,1.5) + Tukey(0.2) +
NormalizationOn, asi que **el anclaje YA es a una imagen Fd-F de 1 onda plana**,
pese a que la variable se llame ``x_das`` y el comentario hable del "piso
DAS_1PW". Cambiar el anclaje "de DAS a Fd-F" no es un experimento: no hay nada
que cambiar.

Lo que SI son dos imagenes distintas (correlacion 0.931 sobre la rejilla comun,
ver verify_fdf1pw_same_grid.py) son las dos rutas Fd-F:

  A) OpHFdf.adjoint      -> el adjunto del operador de medida. Es el anclaje
                            actual y la entrada del modelo.
  B) das_pw(Fd-F, 1 PW)  -> el conformador clasico. Es la fila
                            "Schiffner Fd-F 1 PW" de la tabla canonica.

Anclar a B en vez de a A si seria un experimento real. Pero solo merece la pena
gastar 9 h de GPU si B es MEJOR que A: el anclaje sujeta al modelo hacia la
imagen que se le pasa, asi que anclar a una imagen peor es contraproducente.

Este script responde a eso midiendo A con el mismo protocolo con que ya se
midio B, reutilizando los npy que gen_b1v5_picmus_seeds.py ya guardo en
``npys/das1pw_adjoint/`` (env_log [0,1] sobre la rejilla 612x388: no hace falta
ni GPU ni remuestreo).

Uso
---
    python resultados-finales/scripts/eval_anchor_candidates.py
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATASETS = Path(os.environ.get("TESIS_DATASETS", ROOT / "datos"))  # PICMUS/ y Field II/, ver datos/README.md
sys.path.insert(0, str(ROOT / "evaluacion" / "scripts"))
sys.path.insert(0, str(ROOT / "codigo"))

from eval_and_figs_finales import PHANTOMS  # noqa: E402
from unet_ag.data.rf_loader import PICMUSLoader  # noqa: E402
from unet_ag.eval.evaluate_fieldii import _build_grid  # noqa: E402
from unet_ag.eval.evaluate_picmus import _eval_image  # noqa: E402

NPYS = ROOT / "evaluacion" / "npys"
METR = ROOT / "evaluacion" / "metricas"


def main() -> None:
    rows = []
    for key, mode, phantom, label in PHANTOMS:
        p = NPYS / "das1pw_adjoint" / f"{key}_das1pwadj.npy"
        if not p.exists():
            print(f"FALTA {p}")
            return
        loader = PICMUSLoader(DATASETS / "PICMUS", mode=mode,
                              phantom=phantom, normalize=True)
        gz_t, gx_t = _build_grid(loader[0].geometry, 612, 388)
        gz, gx = gz_t.numpy().astype(np.float64), gx_t.numpy().astype(np.float64)
        img = np.load(p).astype(np.float32)
        met = _eval_image(img, loader.phantom_path, phantom, gz, gx)
        rows.append({"method": "das1pw_adjoint", "seed": "", "phantom": key, **met})
        print(f"{key:16s} {met}")

    out = METR / "metricas_anchor_candidates.csv"
    cols = sorted({k for r in rows for k in r})
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["method", "seed", "phantom"] +
                           [c for c in cols if c not in ("method", "seed", "phantom")])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out}")

    # ── comparacion con la fila canonica de das_pw(Fd-F, 1 PW) ──────────
    canon = {}
    with open(METR / "metricas_picmus_finales.csv") as fh:
        for r in csv.DictReader(fh):
            if r["method"] == "schiffner1pw":
                canon[r["phantom"]] = r

    def g(d, k):
        v = d.get(k, "")
        return float(v) if v not in ("", None) else float("nan")

    print("\n" + "=" * 74)
    print("A = OpHFdf.adjoint (anclaje ACTUAL)   B = das_pw Fd-F 1 PW (fila canonica)")
    print(f"{'metrica':22s} {'A (adjunto)':>12s} {'B (das_pw)':>12s}  {'mejor':>8s}")
    print("-" * 74)
    spec = [("sim_resolution", "fwhm_lat_med_mm", "sim FWHM_lat", True),
            ("exp_resolution", "fwhm_lat_med_mm", "exp FWHM_lat", True),
            ("sim_contrast",   "gcnr",            "sim gCNR",     False),
            ("exp_contrast",   "gcnr",            "exp gCNR",     False),
            ("sim_contrast",   "cnr_db",          "sim CNR dB",   False),
            ("exp_contrast",   "cnr_db",          "exp CNR dB",   False)]
    wins = {"A": 0, "B": 0}
    by_row = {r["phantom"]: r for r in rows}
    for ph, col, lab, lower in spec:
        a = g(by_row.get(ph, {}), col)
        b = g(canon.get(ph, {}), col)
        if a != a or b != b:
            continue
        better = "A" if ((a < b) if lower else (a > b)) else "B"
        wins[better] += 1
        print(f"{lab:22s} {a:12.3f} {b:12.3f}  {better:>8s}")
    print("-" * 74)
    print(f"  A gana {wins['A']}/6   B gana {wins['B']}/6")
    print()
    print("  Si B gana con holgura, anclar a das_pw(Fd-F) en vez de al adjunto")
    print("  es un experimento motivado. Si gana A, el anclaje actual ya es la")
    print("  mejor de las dos y no hay nada que probar.")


if __name__ == "__main__":
    main()

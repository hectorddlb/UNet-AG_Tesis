"""Comprueba las cifras de Field II que cita la tesis sobre partición, rejilla y t_start.

1. Partición determinista de los 300 phantoms de puntos (field_ii_split, semilla
   42): 240 / 30 / 30, y cuántos de cada parte tienen objetivo en
   datos/targets_schiffner_75pw (231 / 28 / 28: el entrenador descarta en
   silencio las muestras sin objetivo, de ahí los 231 pasos por época).
2. Rejilla del entrenador: se construye UNA vez a partir del primer phantom del
   subconjunto de entrenamiento (b1_v5_pulse_aug.py, ``sample0``), de 5 mm a
   (n_s - 1) c / (2 f_s). Los objetivos se conformaron en 5-50 mm: mismo número
   de filas (612), distinta escala.
3. Distribución de t_start · c (en mm) sobre los phantoms de entrenamiento.
4. Dispersión del tiempo inicial entre los 75 ángulos de cada phantom (el
   motivo de usar el t_start de cada ángulo al componer los objetivos).

Uso (desde la raíz del repositorio; Field II en $TESIS_DATASETS):
    python evaluacion/scripts/verify_fieldii_geometria.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[2]
DATASETS = Path(os.environ.get("TESIS_DATASETS", ROOT / "datos"))  # PICMUS/ y Field II/, ver datos/README.md
sys.path.insert(0, str(ROOT / "codigo"))

from unet_ag.data.manifest import field_ii_split  # noqa: E402

NZ = 612


def _scalar(m: dict, key: str) -> float:
    return float(np.asarray(m[key]).ravel()[0])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=Path, default=DATASETS / "field_ii_data_multiangle")
    p.add_argument("--targets", type=Path, default=ROOT / "datos" / "targets_schiffner_75pw")
    p.add_argument("--out", type=Path,
                   default=ROOT / "evaluacion" / "verificacion" / "fieldii_particion_rejilla_tstart.json")
    args = p.parse_args()

    split = field_ii_split(args.data_root, ["points"])["points"]
    has_t = lambda f: (args.targets / "points" / f.stem / "compounded.npy").exists()  # noqa: E731
    parts = {k: getattr(split, k) for k in ("train", "val", "test")}
    counts = {k: dict(phantoms=len(v), con_objetivo=sum(has_t(f) for f in v))
              for k, v in parts.items()}

    # rejilla del entrenador: primer phantom del subconjunto de entrenamiento
    first = parts["train"][0]
    m0 = sio.loadmat(first, variable_names=["rf_data", "fs", "c"])
    ns = int(np.asarray(m0["rf_data"]).shape[0])
    fs, c = _scalar(m0, "fs"), _scalar(m0, "c")
    z_max = (ns - 1) * c / (2.0 * fs)
    dz_e = (z_max - 5e-3) / (NZ - 1)
    dz_t = (50e-3 - 5e-3) / (NZ - 1)
    z45 = 45e-3
    row45 = (z45 - 5e-3) / dz_e
    off45 = row45 * (dz_e - dz_t)

    # t_start * c (angulo 0) en los phantoms de entrenamiento
    shifts, spreads = [], []
    for f in parts["train"]:
        m = sio.loadmat(f, variable_names=["t_start", "angles_deg", "c"])
        ang = np.asarray(m["angles_deg"]).ravel()
        ts = np.asarray(m["t_start"]).ravel()
        t0 = ts[int(np.argmin(np.abs(ang)))]
        shifts.append(float(t0) * _scalar(m, "c") * 1e3)
        spreads.append(float(ts.max() - ts.min()))
    shifts, spreads = np.asarray(shifts), np.asarray(spreads)

    out = dict(
        particion=counts,
        rejilla_entrenador=dict(primer_phantom=first.stem, n_s=ns, fs=fs, c=c,
                                z_max_mm=z_max * 1e3),
        rejilla_objetivos_mm=[5.0, 50.0],
        error_escala_relativo=1.0 - dz_t / dz_e,
        desplazamiento_a_45mm=dict(mm=off45 * 1e3, filas_entrenador=off45 / dz_e,
                                   filas_objetivo=off45 / dz_t),
        t_start_por_c_mm_entrenamiento=dict(n=int(shifts.size), min=float(shifts.min()),
                                            max=float(shifts.max()),
                                            mediana=float(np.median(shifts))),
        dispersion_t_start_entre_angulos=dict(
            mediana_us=float(np.median(spreads) * 1e6),
            min_us=float(spreads.min() * 1e6), max_us=float(spreads.max() * 1e6),
            mediana_mm_axiales=float(np.median(spreads) * c / 2 * 1e3)),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Validación del port en Python del beamformer Fd-F contra das_pw.m.

Compara, sobre el escenario experimental de contraste de PICMUS y la rejilla de
612 x 388 puntos de preparar_entrada.py, la imagen compleja de la implementación
de referencia en MATLAB con la del port (``schiffner_das.das_pw``) ejecutado con
los mismos parámetros: Tukey 0.2, normalización por suma de pesos, banda de
2.25 a 6.75 MHz, índice de inicio nulo, F-number GratingAngleLB(60, 1.5) o
apertura completa, sin antialiasing, en float64.

La referencia solo admite su convención de transmisión nativa (borde, -sin θ),
así que el port se ejecuta en esa misma convención: a 0° coincide con la de
PICMUS; con 75 ángulos la comparación valida la equivalencia numérica, no los
comparadores de 75 ondas planas de la evaluación (que usan la convención corregida).

Métricas: error relativo L2 de la imagen compleja, diferencia máxima relativa al
pico y diferencias de gCNR y CNR en dB entre ambas implementaciones, calculadas
con las MISMAS regiones de interés para las dos: disco interior de radio 0.75 r
en cada inclusión y anillo de fondo de 1.25 r a 1.75 r que excluye las demás
inclusiones, sobre la envolvente en dB recortada a 60 dB y llevada a [0, 1].

Entradas MATLAB: los ``.mat`` de run_das_pw_matlab.m en trabajo/ si existen; si
no, las imágenes ya calculadas en resultados/matlab_<caso>.npy.

Uso (desde la raíz del repositorio; GPU recomendable para 75 ángulos):
    python evaluacion/validacion_matlab/preparar_entrada.py
    python evaluacion/validacion_matlab/comparar_python_matlab.py
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio
import torch

ROOT = Path(__file__).resolve().parents[2]
DATASETS = Path(os.environ.get("TESIS_DATASETS", ROOT / "datos"))  # PICMUS/ y Field II/, ver datos/README.md
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "codigo"))

from schiffner_das import ConstantFNumber, GratingAngleLB, das_pw  # noqa: E402
from schiffner_das.normalizations import NormalizationOn  # noqa: E402
from schiffner_das.windows import Tukey  # noqa: E402

PHANTOM_REL = "experiments/contrast_speckle/contrast_speckle_expe_phantom.hdf5"
DR = 60.0


def to_I(E: np.ndarray) -> np.ndarray:
    e = E / (E.max() + 1e-300)
    return (np.clip(20 * np.log10(e + 1e-300), -DR, 0) + DR) / DR


def gcnr(a: np.ndarray, b: np.ndarray) -> float:
    ha, _ = np.histogram(a, bins=256, range=(0.0, 1.0))
    hb, _ = np.histogram(b, bins=256, range=(0.0, 1.0))
    return float(1.0 - np.minimum(ha / ha.sum(), hb / hb.sum()).sum())


def cnr(a: np.ndarray, b: np.ndarray) -> float:
    return float(20 * np.log10(abs(a.mean() - b.mean()) / np.sqrt(a.var() + b.var())))


def roi_masks(phantom_hdf5: Path, gz_mm: np.ndarray, gx_mm: np.ndarray):
    """Interior (disco 0.75 r) y fondo (anillo 1.25-1.75 r sin otras inclusiones)."""
    g = h5py.File(phantom_hdf5, "r")["US/US_DATASET0000"]
    cx = np.ravel(g["phantom_occlusionCenterX"][()]) * 1e3
    cz = np.ravel(g["phantom_occlusionCenterZ"][()]) * 1e3
    rr = np.ravel(g["phantom_occlusionDiameter"][()]) * 1e3 / 2
    zz, xx = np.meshgrid(gz_mm, gx_mm, indexing="ij")
    d = [np.hypot(zz - z, xx - x) for z, x in zip(cz, cx)]
    inner, ring = np.zeros_like(zz, bool), np.zeros_like(zz, bool)
    for i, r in enumerate(rr):
        other = np.zeros_like(zz, bool)
        for j, r2 in enumerate(rr):
            if j != i:
                other |= d[j] <= 1.25 * r2
        inner |= d[i] <= 0.75 * r
        ring |= (d[i] >= 1.25 * r) & (d[i] <= 1.75 * r) & ~other
    return inner, ring


def port(S: dict, kind: str, ids, device: str) -> np.ndarray:
    F = GratingAngleLB(60.0, 1.5) if kind == "fdf" else ConstantFNumber(0.0)
    gz, gx = np.ravel(S["positions_z"]), np.ravel(S["positions_x"])
    rf, ang = S["data_RF"], np.ravel(S["angles"])
    tot = None
    for a in ids:
        r = das_pw(torch.from_numpy(gx), torch.from_numpy(gz),
                   torch.from_numpy(np.ascontiguousarray(rf[:, :, a])),
                   S["fs"].item(), float(ang[a]), S["width"].item(), S["pitch"].item(),
                   S["c0"].item(), f_bounds=(2.25e6, 6.75e6), index_t0=0,
                   window=Tukey(0.2), F_number=F, normalization=NormalizationOn(),
                   z_chunk=32, device=device, tx_sign=-1.0, tx_reference="edge",
                   precision="float64")
        im = r.image.detach().cpu().numpy()
        tot = im if tot is None else tot + im
    return tot


def matlab_image(case: str, mat_files: list[Path]) -> np.ndarray:
    """Suma de las imágenes de los ``.mat`` de MATLAB, o la ya guardada en resultados/."""
    if mat_files and all(f.exists() for f in mat_files):
        tot = None
        for f in mat_files:
            with h5py.File(f, "r") as h:
                d = h["imgs"][()]
            im = (d["real"] + 1j * d["imag"]) if d.dtype.names else d
            im = np.transpose(im, (0, 2, 1)) if im.ndim == 3 else im.T   # v7.3: orden columna
            s = im.sum(0) if im.ndim == 3 else im
            tot = s if tot is None else tot + s
        return tot
    return np.load(HERE / "resultados" / f"matlab_{case}.npy")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--entrada", type=Path, default=HERE / "trabajo" / "exp_contrast_rf.mat")
    p.add_argument("--picmus", type=Path, default=DATASETS / "PICMUS")
    p.add_argument("--out", type=Path, default=HERE / "resultados" / "validacion_matlab.csv")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device

    S = sio.loadmat(args.entrada)
    gz_mm, gx_mm = np.ravel(S["positions_z"]) * 1e3, np.ravel(S["positions_x"]) * 1e3
    inner, ring = roi_masks(args.picmus / "database" / PHANTOM_REL, gz_mm, gx_mm)
    n_ang = np.ravel(S["angles"]).size
    c0 = int(np.argmin(np.abs(np.ravel(S["angles"]))))
    work = HERE / "trabajo"
    cases = [("fdf_1pw", "fdf", [c0], [work / "matlab_fdf_1pw.mat"]),
             ("das_1pw", "das", [c0], [work / "matlab_das_1pw.mat"]),
             ("fdf_75pw_nativa", "fdf", range(n_ang), sorted(work.glob("matlab_fdf_75_*.mat"))),
             ("das_75pw_nativa", "das", range(n_ang), sorted(work.glob("matlab_das_75_*.mat")))]
    rows = []
    for name, kind, ids, files in cases:
        Mi, Pi = matlab_image(name, files), port(S, kind, ids, device)
        assert Mi.shape == Pi.shape == (gz_mm.size, gx_mm.size), (Mi.shape, Pi.shape)
        IM, IP = to_I(np.abs(Mi)), to_I(np.abs(Pi))
        row = dict(caso=name, n_angulos=len(list(ids)),
                   err_rel_L2=float(np.linalg.norm(Mi - Pi) / np.linalg.norm(Mi)),
                   dif_max_abs=float(np.abs(Mi - Pi).max()),
                   dif_max_rel_al_pico=float(np.abs(Mi - Pi).max() / np.abs(Mi).max()),
                   gcnr_matlab=gcnr(IM[inner], IM[ring]), gcnr_port=gcnr(IP[inner], IP[ring]),
                   cnr_matlab_db=cnr(IM[inner], IM[ring]), cnr_port_db=cnr(IP[inner], IP[ring]))
        row["dif_gcnr"] = row["gcnr_port"] - row["gcnr_matlab"]
        row["dif_cnr_db"] = row["cnr_port_db"] - row["cnr_matlab_db"]
        rows.append(row)
        print(f"{name:16s} err_rel_L2={row['err_rel_L2']:.2e}  "
              f"dif_max/pico={row['dif_max_rel_al_pico']:.2e}  "
              f"|ΔgCNR|={abs(row['dif_gcnr']):.1e}  |ΔCNR|={abs(row['dif_cnr_db']):.1e} dB",
              flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"escrito {args.out}")


if __name__ == "__main__":
    main()

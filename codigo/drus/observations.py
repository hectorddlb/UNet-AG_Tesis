"""Observaciones DDRM: carga RF PICMUS 1PW (0°) + TGC + normalización + By.

Port de ``MATLAB/Results/02_picmus/BH/compute_By.m`` (DRUS-v1). Los parámetros
TGC/escala por phantom vienen de los comentarios del propio script MATLAB:

    tgc alpha : simu_reso=4.5, simu_cont=expe_reso=expe_cont=4.0
    escala    : simu_reso=10, simu_cont=30, expe_reso=10, expe_cont=30
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp

# (acquisition_dir, phantom_dir, tag) por nombre corto del paper
PHANTOMS = {
    "simu_reso": ("simulation", "resolution_distorsion", "simu"),
    "simu_cont": ("simulation", "contrast_speckle", "simu"),
    "expe_reso": ("experiments", "resolution_distorsion", "expe"),
    "expe_cont": ("experiments", "contrast_speckle", "expe"),
}

TGC_ALPHA = {"simu_reso": 4.5, "simu_cont": 4.0, "expe_reso": 4.0, "expe_cont": 4.0}
NORM_SCALE = {"simu_reso": 10.0, "simu_cont": 30.0, "expe_reso": 10.0, "expe_cont": 30.0}

ZERO_DEG_IDX = 37  # transmisión 38 en MATLAB (1-based) = ángulo 0° de 75


@dataclass
class PicmusRF:
    y: np.ndarray            # (D, Ne) — samples × elementos, RF crudo del 0°
    fs: float
    c: float
    initial_time: float
    probe_x: np.ndarray      # (Ne,)
    angle_deg: float


def rf_path(picmus_root: Path, phantom: str) -> Path:
    acq, ph, tag = PHANTOMS[phantom]
    return picmus_root / "database" / acq / ph / f"{ph}_{tag}_dataset_rf.hdf5"


def phantom_from_filename(path) -> str:
    """Nombre corto del fantoma a partir del fichero RF de PICMUS.

    ``resolution_distorsion_expe_dataset_rf.hdf5`` -> ``"expe_reso"``. Lanza
    ``ValueError`` si el nombre no corresponde a ninguno de los cuatro.
    """
    name = Path(path).name
    for short, (_acq, ph, tag) in PHANTOMS.items():
        if name.startswith(f"{ph}_{tag}_"):
            return short
    raise ValueError(f"no se reconoce el fantoma de PICMUS en {name!r}; "
                     f"indícalo explícitamente ({', '.join(PHANTOMS)})")


def load_picmus_rf(picmus_root: Path, phantom: str) -> PicmusRF:
    return load_picmus_rf_file(rf_path(picmus_root, phantom))


def load_picmus_rf_file(path: Path) -> PicmusRF:
    """Lee el RF del ángulo 0° de un fichero ``*_dataset_rf.hdf5`` de PICMUS."""
    with h5py.File(path, "r") as f:
        g = f["US/US_DATASET0000"]
        data = g["data/real"][ZERO_DEG_IDX]          # (Ne, D)
        ang = float(np.degrees(g["angles"][ZERO_DEG_IDX]))
        fs = float(np.asarray(g["sampling_frequency"]).ravel()[0])
        c = float(np.asarray(g["sound_speed"]).ravel()[0])
        t0 = float(np.asarray(g["initial_time"]).ravel()[0])
        probe = g["probe_geometry"][:]
    assert abs(ang) < 0.1, f"transmisión {ZERO_DEG_IDX} no es 0°: {ang}°"
    return PicmusRF(
        y=np.asarray(data, dtype=np.float64).T,      # (D, Ne)
        fs=fs, c=c, initial_time=t0,
        probe_x=np.asarray(probe[0], dtype=np.float64),
        angle_deg=ang,
    )


def compute_by(
    B: sp.csr_matrix,
    rf: PicmusRF,
    big_k: int,
    k_min: int,
    phantom: str,
    tgc_alpha: float | None = None,
    norm_scale: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (By, y_procesado). Réplica de la sección Input de compute_By.m.

    ``tgc_alpha`` y ``norm_scale`` toman por omisión los valores del fantoma;
    darlos explícitamente permite aplicar la cadena a datos propios.
    """
    alpha = TGC_ALPHA[phantom] if tgc_alpha is None else float(tgc_alpha)
    scale = NORM_SCALE[phantom] if norm_scale is None else float(norm_scale)
    y = rf.y[:big_k, :].copy()                        # (K, Ne)
    if y.shape[0] < big_k:
        raise ValueError(f"RF de {phantom} tiene {rf.y.shape[0]} samples < K={big_k}")

    # TGC exponencial + clamp del near-field (primeros k_min+150 samples)
    t = rf.initial_time + np.arange(big_k) / rf.fs
    tgc = np.exp(alpha * t / t.max())
    tgc[: k_min + 150] = tgc[k_min + 150]
    y *= tgc[:, None]

    # Normalización global + escala por phantom
    y /= np.max(np.abs(y))
    y *= scale

    by = B @ y.flatten(order="F")                     # (:) de MATLAB = column-major
    return by.astype(np.float64), y

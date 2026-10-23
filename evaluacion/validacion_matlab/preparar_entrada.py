"""Entrada común para la validación MATLAB del beamformer Fd-F.

Empaqueta en un ``.mat`` el RF de los 75 ángulos del escenario experimental de
contraste de PICMUS, su geometría y la rejilla de la comparación (612 x 388
puntos, z de 5 a 50 mm y x de ±19.2 mm), de modo que ``das_pw.m`` y el port en
Python lean exactamente los mismos datos.

Uso (desde la raíz del repositorio):
    python evaluacion/validacion_matlab/preparar_entrada.py
    -> evaluacion/validacion_matlab/trabajo/exp_contrast_rf.mat   (~255 MB, no versionado)
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[2]
DATASETS = Path(os.environ.get("TESIS_DATASETS", ROOT / "datos"))  # PICMUS/ y Field II/, ver datos/README.md
sys.path.insert(0, str(ROOT / "codigo"))

from schiffner_das import load_picmus_hdf5  # noqa: E402

RF_REL = "experiments/contrast_speckle/contrast_speckle_expe_dataset_rf.hdf5"
GZ = np.linspace(5e-3, 50e-3, 612)
GX = np.linspace(-128 * 300e-6 / 2, 128 * 300e-6 / 2, 388)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--picmus", type=Path, default=DATASETS / "PICMUS")
    p.add_argument("--out", type=Path,
                   default=Path(__file__).resolve().parent / "trabajo" / "exp_contrast_rf.mat")
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    ds = load_picmus_hdf5(args.picmus / "database" / RF_REL)
    rf = np.stack([ds.rf_for_angle(a).numpy() for a in range(len(ds.steering_angles_rad))],
                  -1).astype(np.float64)
    sio.savemat(args.out, dict(
        data_RF=rf, angles=np.asarray(ds.steering_angles_rad, float), fs=float(ds.f_s),
        pitch=float(ds.element_pitch), width=float(ds.element_width), c0=float(ds.c_avg),
        positions_z=GZ, positions_x=GX, t0=float(ds.initial_time_s)), do_compression=False)
    print(f"escrito {args.out}: RF {rf.shape}, ángulo central (base 1) = "
          f"{int(np.argmin(np.abs(np.asarray(ds.steering_angles_rad)))) + 1}")


if __name__ == "__main__":
    main()

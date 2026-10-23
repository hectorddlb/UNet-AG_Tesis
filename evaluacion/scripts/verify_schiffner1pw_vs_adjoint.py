"""Comprueba si "Schiffner Fd-F 1 PW" es la entrada de UNet-AG.

MOTIVO
------
Hay dos rutas de codigo que aplican la MISMA apodizacion Fd-F a UNA sola onda
plana, y el proyecto las trata como cosas distintas:

  A) ``schiffner_das.das_pw(1 angulo, GratingAngleLB(60,1.5), Tukey(0.2),
     NormalizationOn)``  -> el metodo clasico, protocolo canonico.
  B) ``OpHFdf.adjoint``  -> el adjunto del operador de medida, cuyos DEFAULTS
     son exactamente GratingAngleLB(60,1.5) + Tukey(0.2) + NormalizationOn.
     Es lo que produce ``x0``, la ENTRADA de UNet-AG, y por tanto --con el
     residual de ``out_conv`` inicializada a cero-- su salida en la epoca 0.

Si A y B coinciden, la fila "Schiffner Fd-F 1 PW" de la tabla no es solo un
cuarto metodo de 1 PW: es la linea base exacta del modelo propuesto, y la
comparacion UNet-AG vs esa fila mide lo que aporta la RED, separado de lo que
aporta la apodizacion.

Ademas aclara una ambiguedad real del proyecto: la columna ``das_1pw`` de los
``eval_picmus.csv`` por corrida se calcula con la ruta B (adjunto, Fd-F),
mientras que ``das1pw`` de la tabla canonica usa ``ConstantFNumber(0)``
--apertura completa--. Son dos conformadores distintos con la misma etiqueta,
y este script cuantifica cuanto se separan.

QUE COMPARA
-----------
Las tres imagenes, todas llevadas al protocolo canonico (env_log [0,1] sobre
la rejilla de evaluacion 612x388):

  schiffner1pw   npys/schiffner1pw/<key>_schiffner1pw.npy   (complejo, scan grid)
  adjunto        npys/das1pw_adjoint/<key>_das1pwadj.npy    (ya env_log 612x388)
  das1pw         npys/das1pw/<key>_das1pw.npy               (complejo, scan grid)

Uso
---
    python resultados-finales/scripts/verify_schiffner1pw_vs_adjoint.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATASETS = Path(os.environ.get("TESIS_DATASETS", ROOT / "datos"))  # PICMUS/ y Field II/, ver datos/README.md
sys.path.insert(0, str(ROOT / "evaluacion" / "scripts"))
sys.path.insert(0, str(ROOT / "codigo"))

from eval_and_figs_finales import (  # noqa: E402
    PHANTOMS,
    env_log01,
    interp_to,
    load_scan_axes,
)
from unet_ag.data.rf_loader import PICMUSLoader  # noqa: E402
from unet_ag.eval.evaluate_fieldii import _build_grid  # noqa: E402

NPYS = ROOT / "evaluacion" / "npys"


def to_eval_grid(path: Path, key: str, gz: np.ndarray, gx: np.ndarray) -> np.ndarray:
    """Complejo en scan grid -> env_log [0,1] en la rejilla de evaluacion."""
    src_z, src_x = load_scan_axes(key)
    return interp_to(env_log01(np.load(path)), src_z, src_x, gz, gx)


def stats(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    return {
        "corr": float(np.corrcoef(a.ravel(), b.ravel())[0, 1]),
        "rmse": float(np.sqrt(np.mean(d ** 2))),
        "max_abs": float(np.abs(d).max()),
        "media_a": float(a.mean()),
        "media_b": float(b.mean()),
    }


def main() -> None:
    print("Comparacion sobre la rejilla canonica de evaluacion (612x388, env_log)\n")
    filas = []
    for key, mode, phantom, label in PHANTOMS:
        loader = PICMUSLoader(DATASETS / "PICMUS", mode=mode,
                              phantom=phantom, normalize=True)
        gz_t, gx_t = _build_grid(loader[0].geometry, 612, 388)
        gz, gx = gz_t.numpy().astype(np.float64), gx_t.numpy().astype(np.float64)

        p_s1 = NPYS / "schiffner1pw" / f"{key}_schiffner1pw.npy"
        p_adj = NPYS / "das1pw_adjoint" / f"{key}_das1pwadj.npy"
        p_das = NPYS / "das1pw" / f"{key}_das1pw.npy"
        for p in (p_s1, p_adj, p_das):
            if not p.exists():
                print(f"FALTA {p} — ejecuta antes gen_classical_picmus_fixed.py")
                return

        s1 = to_eval_grid(p_s1, key, gz, gx)
        das = to_eval_grid(p_das, key, gz, gx)
        adj = np.load(p_adj).astype(np.float32)

        a = stats(s1, adj)   # las dos rutas Fd-F de 1 PW
        b = stats(s1, das)   # Fd-F frente a apertura completa
        filas.append((label, a, b))

        print(f"=== {label} ({key}) ===")
        print(f"  das_pw(Fd-F,1PW)  vs  OpHFdf.adjoint : "
              f"corr={a['corr']:.4f}  rmse={a['rmse']:.4f}  max|d|={a['max_abs']:.4f}")
        print(f"  das_pw(Fd-F,1PW)  vs  DAS 1 PW (F=0) : "
              f"corr={b['corr']:.4f}  rmse={b['rmse']:.4f}  max|d|={b['max_abs']:.4f}")
        print()

    print("=" * 72)
    print("LECTURA")
    print("  Si corr(Fd-F, adjunto) ~ 1 y su rmse << rmse(Fd-F, F=0), las dos")
    print("  rutas son el mismo conformador y la entrada de UNet-AG ES")
    print("  Schiffner Fd-F 1 PW. Si no, son metodos distintos y hay que")
    print("  reportarlos por separado.")
    m_adj = float(np.mean([a["rmse"] for _l, a, _b in filas]))
    m_das = float(np.mean([b["rmse"] for _l, _a, b in filas]))
    c_adj = float(np.mean([a["corr"] for _l, a, _b in filas]))
    print(f"\n  media rmse(Fd-F, adjunto)   = {m_adj:.4f}   corr media = {c_adj:.4f}")
    print(f"  media rmse(Fd-F, DAS F=0)   = {m_das:.4f}")
    if m_das > 0:
        print(f"  razon                        = {m_das / max(m_adj, 1e-12):.1f}x")


if __name__ == "__main__":
    main()

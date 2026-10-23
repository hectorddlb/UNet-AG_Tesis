"""Barrido de beta {0.3..0.7} medido con el PROTOCOLO CANONICO de la tesis.

CONTEXTO
--------
La eleccion de beta se sostiene en una curva sobre {0.3, 0.4, 0.5, 0.6, 0.7}.
Los CSV que ya existen de ese barrido
(``results_v2/b1_v5_*/eval_picmus.csv``) vienen del evaluador POR CORRIDA, cuya
referencia "DAS 1 PW" es el adjunto Fd-F y no el DAS de apertura completa de la
tabla canonica. Poner esa curva junto a la tabla principal mezclaria las dos
fuentes justo en la figura que sostiene la eleccion de beta.

Como los cinco ``best.pt`` siguen intactos, este script vuelve a inferir los 4
fantomas PICMUS con cada uno y los mide con ``_eval_image`` sobre la rejilla
612x388 -- exactamente la misma cadena que ``eval_and_figs_finales.py``.

Las cinco corridas comparten configuracion (20 epocas, seed 42, points,
mse=0.7 l1=0.2 ssim=0.1, pulse aug prob=0.7); la unica variable es anchor_w.
Verificado en los logs de entrenamiento 192071 / 198399 / 193078 / 198402 /
198405.

CONTROL DE CONSISTENCIA
-----------------------
beta=0.50 es el modelo principal de la tesis, asi que su fila DEBE reproducir
la fila ``b1v5,s42`` de ``metricas_picmus_finales.csv``. El script lo comprueba
y aborta con codigo != 0 si no coincide dentro de tolerancia: si esa fila no
sale igual, la cadena no es la canonica y la curva no es publicable.

Uso
---
    python resultados-finales/scripts/eval_beta_sweep_canonico.py --device cuda
"""
from __future__ import annotations

import argparse
import os
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DATASETS = Path(os.environ.get("TESIS_DATASETS", ROOT / "datos"))  # PICMUS/ y Field II/, ver datos/README.md
sys.path.insert(0, str(ROOT / "evaluacion" / "scripts"))
sys.path.insert(0, str(ROOT / "codigo"))

from eval_and_figs_finales import PHANTOMS  # noqa: E402
from unet_ag.data.rf_loader import PICMUSLoader  # noqa: E402
from unet_ag.eval.evaluate_fieldii import (  # noqa: E402
    _build_grid,
    _load_b1_v3_ckpt,
    _reconstruct_b1,
)
from unet_ag.eval.evaluate_picmus import _eval_image  # noqa: E402

# beta -> subdirectorio de results_v2 (mapeo tomado de los logs de train)
SWEEP = {
    0.30: "b1_v5_pulse",
    0.40: "b1_v5_anchor04",
    0.50: "b1_v5_anchor05",
    0.60: "b1_v5_anchor06",
    0.70: "b1_v5_anchor07",
}

NPYS = ROOT / "evaluacion" / "npys" / "beta_sweep"
METR = ROOT / "evaluacion" / "metricas"

# Fila canonica de b1v5 seed 42 (metricas_picmus_finales.csv) para el control.
CANON_S42 = {
    ("sim_resolution", "fwhm_lat_med_mm"): 1.375744104385376,
    ("exp_resolution", "fwhm_lat_med_mm"): 2.7551991939544678,
    ("sim_contrast",   "gcnr"):            0.5576750210072512,
    ("exp_contrast",   "gcnr"):            0.7902109395544537,
}
TOL = 1e-3


def main() -> int:
    p = argparse.ArgumentParser()
    # Los 5 checkpoints del barrido NO se incluyen en el repositorio (203 MB
    # de modelos intermedios). El CSV resultante si esta, en
    # evaluacion/metricas/metricas_beta_sweep.csv, y basta para redibujar la
    # figura. Para regenerarlo hay que reentrenar el barrido o apuntar
    # --sweep_root a un directorio con los best.pt.
    p.add_argument("--sweep_root", type=Path,
                   default=ROOT / "resultados" / "beta_sweep")
    p.add_argument("--picmus", type=Path, default=DATASETS / "PICMUS")
    p.add_argument("--nz", type=int, default=612)
    p.add_argument("--nx", type=int, default=388)
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    print(f"device = {device}   sweep_root = {args.sweep_root}")

    missing = [n for n in SWEEP.values()
               if not (args.sweep_root / n / "checkpoints" / "best.pt").exists()]
    if missing:
        print(f"ERROR: faltan checkpoints: {missing}")
        return 2

    # Los loaders y las rejillas no dependen de beta: se construyen una vez.
    ctx = {}
    for key, mode, phantom, _label in PHANTOMS:
        loader = PICMUSLoader(args.picmus, mode=mode, phantom=phantom, normalize=True)
        sample = loader[0]
        gz_t, gx_t = _build_grid(sample.geometry, args.nz, args.nx)
        ctx[key] = (loader, sample, gz_t, gx_t, phantom)

    rows = []
    for beta, run in SWEEP.items():
        ckpt = args.sweep_root / run / "checkpoints" / "best.pt"
        model = _load_b1_v3_ckpt(ckpt, device, base_ch=32)
        out_dir = NPYS / f"beta{int(round(beta * 100)):03d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n== beta={beta:.2f}  run={run}")
        for key, _mode, _ph, _lab in PHANTOMS:
            loader, sample, gz_t, gx_t, phantom = ctx[key]
            t0 = time.perf_counter()
            pred_env, _das_env = _reconstruct_b1(model, sample, gz_t, gx_t, device)
            np.save(out_dir / f"{key}_b1v5.npy", pred_env.astype(np.float32))
            met = _eval_image(pred_env, loader.phantom_path, phantom,
                              gz_t.numpy().astype(np.float64),
                              gx_t.numpy().astype(np.float64))
            rows.append({"beta": f"{beta:.2f}", "run": run, "phantom": key, **met})
            print(f"  {key:16s} {time.perf_counter() - t0:6.1f}s  "
                  + "  ".join(f"{k}={v:.4f}" for k, v in met.items()
                              if isinstance(v, float) and v == v))
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    METR.mkdir(parents=True, exist_ok=True)
    out_csv = METR / "metricas_beta_sweep.csv"
    cols = sorted({k for r in rows for k in r} - {"beta", "run", "phantom"})
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["beta", "run", "phantom"] + cols)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {out_csv}")

    # ── tabla resumen ───────────────────────────────────────────────────
    by = {(r["beta"], r["phantom"]): r for r in rows}

    def g(beta, ph, col):
        v = by.get((f"{beta:.2f}", ph), {}).get(col, "")
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("nan")

    print("\n" + "=" * 78)
    print("Barrido de beta — protocolo canonico (env_log 612x388, _eval_image)")
    print(f"{'beta':>6s} {'sim FWHM_lat':>13s} {'exp FWHM_lat':>13s} "
          f"{'sim gCNR':>10s} {'exp gCNR':>10s} {'sim CNR dB':>11s} {'exp CNR dB':>11s}")
    print("-" * 78)
    for beta in SWEEP:
        print(f"{beta:6.2f} {g(beta,'sim_resolution','fwhm_lat_med_mm'):13.3f} "
              f"{g(beta,'exp_resolution','fwhm_lat_med_mm'):13.3f} "
              f"{g(beta,'sim_contrast','gcnr'):10.3f} "
              f"{g(beta,'exp_contrast','gcnr'):10.3f} "
              f"{g(beta,'sim_contrast','cnr_db'):11.3f} "
              f"{g(beta,'exp_contrast','cnr_db'):11.3f}")
    print("-" * 78)
    print("  Referencia canonica DAS 1 PW: sim 1.236 / exp 3.728 / "
          "gCNR 0.558 / 0.528 / CNR 0.744 / -3.974")

    # ── control de consistencia contra la fila canonica de seed 42 ──────
    print("\nControl: beta=0.50 debe reproducir la fila canonica b1v5,s42")
    bad = []
    for (ph, col), ref in CANON_S42.items():
        got = g(0.50, ph, col)
        ok = abs(got - ref) <= TOL
        print(f"  {ph:16s} {col:18s} obtenido={got:.6f}  canonico={ref:.6f}  "
              f"{'OK' if ok else 'DISCREPA'}")
        if not ok:
            bad.append((ph, col, got, ref))
    if bad:
        print("\nFALLO: la cadena no reproduce la fila canonica. La curva NO es")
        print("publicable junto a la tabla principal hasta resolver esto.")
        return 1
    print("\nOK — misma cadena que la tabla canonica. Curva publicable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

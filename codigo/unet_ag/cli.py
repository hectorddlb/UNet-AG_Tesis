# -*- coding: utf-8 -*-
"""Interfaz de línea de órdenes de UNet-AG.

    unetag info
    unetag reconstruct --picmus <fichero.hdf5> --out bmode.png
    unetag reconstruct --rf datos.npy --geometry geo.json --out bmode.png

Tras `pip install .` la orden `unetag` queda disponible en el PATH.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .inference import DYN_RANGE_DB, Geometry, UNetAG, load_picmus_rf, to_decibels


def _save_png(bmode: np.ndarray, out: Path, gz_mm=None, gx_mm=None,
              title: str | None = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    db = to_decibels(bmode)
    fig, ax = plt.subplots(figsize=(4.6, 5.4))
    if gz_mm is not None and gx_mm is not None:
        n = int(np.searchsorted(gz_mm, 55.0, side="right"))
        im = ax.imshow(db[:n], cmap="gray", vmin=-DYN_RANGE_DB, vmax=0.0,
                       aspect="equal",
                       extent=[gx_mm[0], gx_mm[-1], gz_mm[n - 1], gz_mm[0]])
        ax.set_xlabel("Lateral (mm)")
        ax.set_ylabel("Profundidad (mm)")
    else:
        im = ax.imshow(db, cmap="gray", vmin=-DYN_RANGE_DB, vmax=0.0)
        ax.set_xticks([]); ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03).set_label("dB")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _cmd_info(args) -> int:
    try:
        m = UNetAG.from_pretrained(seed=args.seed, device=args.device)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1
    print(f"UNet-AG — reconstrucción B-mode desde 1 onda plana")
    print(f"  checkpoint : {m.checkpoint}")
    print(f"  semilla    : {args.seed}")
    print(f"  parámetros : {m.n_parameters:,}")
    print(f"  dispositivo: {m.device}")
    print(f"  rejilla    : {m.nz} x {m.nx}")
    print(f"  ganancia γ : {float(m.model.gain):.4f}")
    return 0


def _cmd_reconstruct(args) -> int:
    if not args.picmus and not args.rf:
        print("Indica --picmus <hdf5> o --rf <npy> --geometry <json>",
              file=sys.stderr)
        return 2
    try:
        model = UNetAG(args.ckpt, device=args.device) if args.ckpt \
            else UNetAG.from_pretrained(seed=args.seed, device=args.device)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1

    if args.picmus:
        rf, geom = load_picmus_rf(args.picmus, angle_index=args.angle_index)
        etiqueta = Path(args.picmus).stem
    else:
        rf = np.load(args.rf)
        if args.geometry:
            geom = Geometry(**json.loads(Path(args.geometry).read_text()))
        else:
            print("Aviso: sin --geometry se asume la geometría de PICMUS "
                  "(pitch 300 µm, fs 20.832 MHz, c 1540 m/s).", file=sys.stderr)
            geom = Geometry(n_elements=rf.shape[1], n_samples=rf.shape[0],
                            pitch_m=300e-6, sampling_freq_hz=20.832e6)
        etiqueta = Path(args.rf).stem

    print(f"Reconstruyendo {etiqueta}  "
          f"(RF {rf.shape[0]}x{rf.shape[1]}, {model.device}) ...")
    bmode, adjunto = model.reconstruct(rf, geom, return_adjoint=True)

    out = Path(args.out)
    if out.suffix.lower() == ".npy":
        np.save(out, bmode)
    else:
        gz, gx = model.grid(geom)
        _save_png(bmode, out, gz * 1e3, gx * 1e3, title=f"UNet-AG — {etiqueta}")
    print(f"  escrito {out}")

    if args.save_adjoint:
        a = Path(args.save_adjoint)
        if a.suffix.lower() == ".npy":
            np.save(a, adjunto)
        else:
            gz, gx = model.grid(geom)
            _save_png(adjunto, a, gz * 1e3, gx * 1e3,
                      title=f"Adjunto Fd-F — {etiqueta}")
        print(f"  escrito {a} (solución inicial)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="unetag",
        description="UNet-AG: reconstrucción B-mode desde una sola onda plana.")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = dict(seed="s42", device="auto")
    pi = sub.add_parser("info", help="muestra el modelo cargado")
    pi.add_argument("--seed", default=common["seed"], choices=["s42", "s1", "s2"])
    pi.add_argument("--device", default=common["device"])
    pi.set_defaults(func=_cmd_info)

    pr = sub.add_parser("reconstruct", help="reconstruye una imagen B-mode")
    src = pr.add_argument_group("entrada")
    src.add_argument("--picmus", type=Path, help="fichero RF de PICMUS (.hdf5)")
    src.add_argument("--rf", type=Path, help="RF crudo (n_samples, n_elements) .npy")
    src.add_argument("--geometry", type=Path, help="geometría en JSON (con --rf)")
    src.add_argument("--angle-index", type=int, default=None,
                     help="índice de ángulo en PICMUS (por omisión, el de 0°)")
    pr.add_argument("--out", type=Path, default=Path("bmode.png"),
                    help="salida .png o .npy (por omisión bmode.png)")
    pr.add_argument("--save-adjoint", type=Path, default=None,
                    help="guarda además la solución inicial (adjunto Fd-F)")
    pr.add_argument("--ckpt", type=Path, default=None,
                    help="checkpoint alternativo")
    pr.add_argument("--seed", default=common["seed"], choices=["s42", "s1", "s2"])
    pr.add_argument("--device", default=common["device"])
    pr.set_defaults(func=_cmd_reconstruct)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

# -*- coding: utf-8 -*-
"""Interfaz de línea de órdenes de la réplica DRUS.

    drus info
    drus prepare operador --picmus <raíz de PICMUS>
    drus prepare svd
    drus prepare prior
    drus reconstruct --picmus <fichero.hdf5> --out bmode.png

Tras `pip install ".[drus]"` la orden `drus` queda disponible en el PATH. Los
artefactos (operador, SVD y prior) se localizan con --artifacts o con la
variable DRUS_ARTIFACTS; ver `drus.preparacion`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .inference import DRUS, DYN_RANGE_DB, THESIS_GAMMA
from .observations import PHANTOMS, phantom_from_filename
from .preparacion import (GRID_NX, GRID_NZ, GRID_X_MM, GRID_Z_MM,
                          build_operator, download_prior, picmus_root_from,
                          resolve_artifacts, stage_bh, stage_svd)


def _size_gib(p: Path) -> float:
    if p.is_file():
        return p.stat().st_size / 2**30
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 2**30
    return 0.0


def _save_png(res, out: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    db = res.bmode * DYN_RANGE_DB - DYN_RANGE_DB
    fig, ax = plt.subplots(figsize=(4.8, 5.0))
    im = ax.imshow(db, cmap="gray", vmin=-DYN_RANGE_DB, vmax=0.0, aspect="equal",
                   extent=[res.x_axis[0] * 1e3, res.x_axis[-1] * 1e3,
                           res.z_axis[-1] * 1e3, res.z_axis[0] * 1e3])
    ax.set_xlabel("Lateral (mm)")
    ax.set_ylabel("Profundidad (mm)")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03).set_label("dB")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _cmd_info(args) -> int:
    try:
        art = resolve_artifacts(args.artifacts, require=True)
    except FileNotFoundError:
        art = resolve_artifacts(args.artifacts, require=False)
    st = art.status()
    print("DRUS — réplica del método de difusión (Zhang et al., 2023)")
    print(f"  artefactos : {art.root}")
    for name, path in (("operador", art.operador), ("svd", art.svd),
                       ("prior", art.prior)):
        mark = "ok " if st[name] else "falta"
        print(f"    [{mark}] {name:9s} {path}  ({_size_gib(path):.1f} GiB)")
    print(f"  rejilla    : {GRID_NZ} x {GRID_NX}, x ∈ {list(GRID_X_MM)} mm, "
          f"z ∈ {list(GRID_Z_MM)} mm")
    print("  γ de la tesis por fantoma (semilla 0, prior sin ajuste fino):")
    for k, g in THESIS_GAMMA.items():
        print(f"    {k:10s} γ = {g:g}")
    if art.missing():
        print("\nFaltan artefactos. Ver `drus prepare --help`.")
        return 1
    return 0


def _cmd_prepare(args) -> int:
    art = resolve_artifacts(args.artifacts, require=False)
    if args.what == "operador":
        build_operator(picmus_root_from(args.picmus), art.operador)
    elif args.what == "svd":
        if not (art.operador / "B.npz").exists():
            print(f"Falta el operador en {art.operador}: "
                  f"ejecuta antes `drus prepare operador`.", file=sys.stderr)
            return 1
        if args.stage in ("bh", "all"):
            stage_bh(art.operador, art.svd)
        if args.stage in ("svd", "all"):
            stage_svd(art.operador, art.svd)
            if not args.keep_bh:
                (art.svd / "BH.npy").unlink(missing_ok=True)
    elif args.what == "prior":
        download_prior(art.prior)
    print(f"  artefactos en {art.root}")
    return 0


def _cmd_reconstruct(args) -> int:
    try:
        model = DRUS.from_artifacts(args.artifacts, device=args.device)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 1
    phantom = args.phantom or phantom_from_filename(args.picmus)
    print(f"Reconstruyendo {Path(args.picmus).name} ({phantom}, "
          f"γ = {args.gamma if args.gamma is not None else THESIS_GAMMA[phantom]:g}, "
          f"semilla {args.seed}, {model.device}) ...", flush=True)
    res = model.reconstruct_picmus(args.picmus, phantom=phantom, gamma=args.gamma,
                                   seed=args.seed)
    out = Path(args.out)
    if out.suffix.lower() == ".npy":
        np.save(out, res.bmode)
    else:
        _save_png(res, out, f"DRUS — {phantom} (γ = {res.gamma:g})")
    print(f"  escrito {out}  ({res.seconds:.0f} s de muestreo)")
    if args.save_x:
        np.save(args.save_x, res.x)
        print(f"  escrito {args.save_x} (salida con signo, dominio del prior)")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="drus",
        description="DRUS: reconstrucción B-mode por difusión (DDRM) desde una onda plana.")
    p.add_argument("--artifacts", type=Path, default=None,
                   help="raíz de artefactos (por omisión $DRUS_ARTIFACTS)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("info", help="estado de los artefactos y parámetros de la tesis")
    pi.set_defaults(func=_cmd_info)

    pp = sub.add_parser("prepare", help="construye los artefactos (una sola vez)")
    pp.add_argument("what", choices=["operador", "svd", "prior"])
    pp.add_argument("--picmus", type=Path, default=None,
                    help="raíz de PICMUS (por omisión $TESIS_DATASETS/PICMUS)")
    pp.add_argument("--stage", choices=["bh", "svd", "all"], default="all",
                    help="svd: etapas a ejecutar (en clúster conviene separarlas)")
    pp.add_argument("--keep-bh", action="store_true",
                    help="svd: conserva BH.npy (17 GB) al terminar")
    pp.set_defaults(func=_cmd_prepare)

    pr = sub.add_parser("reconstruct", help="reconstruye un fantoma de PICMUS")
    pr.add_argument("--picmus", type=Path, required=True,
                    help="fichero RF de PICMUS (*_dataset_rf.hdf5)")
    pr.add_argument("--phantom", choices=list(PHANTOMS), default=None,
                    help="fantoma (por omisión se deduce del nombre del fichero)")
    pr.add_argument("--gamma", type=float, default=None,
                    help="γ del muestreador (por omisión, el de la tesis)")
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--out", type=Path, default=Path("bmode_drus.png"),
                    help="salida .png o .npy (envolvente log en [0, 1])")
    pr.add_argument("--save-x", type=Path, default=None,
                    help="guarda además la salida con signo (.npy)")
    pr.add_argument("--device", default="auto")
    pr.set_defaults(func=_cmd_reconstruct)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

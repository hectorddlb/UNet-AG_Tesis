"""Fase F.2-A — Uᵀ·By para los 600 phantoms Field II (targets teacher DRUS).

Espejo escalado del bloque final de ``run_svd_bh.stage_svd``: computa la
observación ``By`` de cada phantom (via ``compute_by_fieldii``, misma
alineación t_start + TGC + pad_extra que la calibración de Fase F.1) y la
proyecta a la base singular izquierda, ``Uty = Uᵀ·By`` — lo único que el
sampler DDRM necesita de U (17 GB). Se corre una sola vez en C0 (128 GB) y
deja los ``Uty_<phantom>.npy`` en el dir de Fase B junto a S/Vt.

Gate G-F2a: reconstruye ``Uty_points_0001`` y lo compara con el guardado por
la Fase B (rel_err < 1e-4). Si falla, aborta antes de escribir los 600.

Uso (dentro del sbatch, en C0):
    python zhang_drus/scripts/make_uty_fieldii.py \
        --data_root "datasets/Field II/field_ii_data_multiangle" \
        --phase_a_dir zhang_drus/results_drus/fieldii_phaseA \
        --svd_dir     zhang_drus/results_drus/fieldii_phaseB
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(_ROOT / "unet_ag"))

from build_h_fieldii import load_mat
from drus.physics_fieldii import compute_by_fieldii


def list_phantoms(data_root: Path) -> list[str]:
    ids: list[str] = []
    for ptype in ("points", "cysts"):
        ids += sorted(p.stem for p in (data_root / ptype).glob("*.mat"))
    return ids


def compute_by_stack(data_root: Path, phase_a_dir: Path, phantoms: list[str]):
    meta = json.loads((phase_a_dir / "meta.json").read_text())
    B = sp.load_npz(phase_a_dir / "B.npz").tocsr()
    npix = B.shape[0]
    fs, big_k, k_min, pad = meta["fs"], meta["K"], meta["k_min"], meta["pad_extra"]
    stack = np.empty((npix, len(phantoms)), dtype=np.float32)
    t0 = time.time()
    for j, pid in enumerate(phantoms):
        m = load_mat(data_root, pid)
        by, _ = compute_by_fieldii(B, m["rf0"], m["t_start0"], fs, big_k, k_min, pad_extra=pad)
        stack[:, j] = by.astype(np.float32)
        if j % 50 == 0 or j == len(phantoms) - 1:
            print(f"[by] {j + 1}/{len(phantoms)} ({pid})  {time.time() - t0:.0f}s", flush=True)
    return stack


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=Path, required=True)
    ap.add_argument("--phase_a_dir", type=Path, required=True, help="B.npz + meta.json")
    ap.add_argument("--svd_dir", type=Path, required=True, help="U.npy (in) + Uty_*.npy (out)")
    ap.add_argument("--out_dir", type=Path, default=None, help="default = svd_dir")
    args = ap.parse_args()
    out_dir = args.out_dir or args.svd_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    phantoms = list_phantoms(args.data_root)
    print(f"[uty] {len(phantoms)} phantoms | svd_dir={args.svd_dir}", flush=True)

    by_stack = compute_by_stack(args.data_root, args.phase_a_dir, phantoms)   # (N, P) f32

    t0 = time.time()
    print("[uty] cargando U.npy (17 GB) a RAM …", flush=True)
    U = np.load(args.svd_dir / "U.npy")                                        # (N, N) f32
    print(f"[uty] U {U.shape} en RAM ({time.time() - t0:.0f}s) → sgemm Uᵀ·By …", flush=True)
    uty = U.T @ by_stack                                                       # (N, P) f32
    print(f"[uty] Uᵀ·By listo ({time.time() - t0:.0f}s)", flush=True)

    # ── Gate G-F2a: consistencia con el Uty guardado por Fase B ──
    ref_p = args.svd_dir / "Uty_points_0001.npy"
    if ref_p.exists() and "points_0001" in phantoms:
        k = phantoms.index("points_0001")
        ref = np.load(ref_p).astype(np.float32)
        rel = float(np.linalg.norm(uty[:, k] - ref) / (np.linalg.norm(ref) + 1e-30))
        status = "PASS" if rel < 1e-4 else "FAIL"
        print(f"[G-F2a] rel_err(Uty_points_0001 vs FaseB) = {rel:.3e} | {status}", flush=True)
        if status == "FAIL":
            raise SystemExit(f"G-F2a FAIL (rel={rel:.3e}) — no se escriben los 600")

    for j, pid in enumerate(phantoms):
        np.save(out_dir / f"Uty_{pid}.npy", np.ascontiguousarray(uty[:, j]))
    (out_dir / "uty_manifest.json").write_text(json.dumps(
        {"n_phantoms": len(phantoms), "phantoms": phantoms}, indent=2))
    print(f"[done] {len(phantoms)} Uty escritos en {out_dir}", flush=True)


if __name__ == "__main__":
    main()

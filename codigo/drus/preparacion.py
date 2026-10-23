"""Artefactos de DRUS: dónde viven y cómo se construyen.

DRUS necesita tres artefactos que no caben en el repositorio. Se guardan bajo
un mismo directorio raíz con esta forma::

    <artefactos>/
    ├── operador/   H.npz, B.npz, grid.npz, meta.json, By_<fantoma>.npy   (~0.3 GB)
    ├── svd/        S.npy, U.npy, Vt.npy, Uty_<fantoma>.npy, svd_meta.json (~34 GB)
    └── prior/      256x256_diffusion_uncond.pt                           (2.2 GB)

- ``operador``: modelo físico de PICMUS (H y el conformador apodizado B) sobre
  la rejilla de la tesis, 256×256 con x ∈ [−18, 18] mm y z ∈ [9, 48] mm, y las
  observaciones ``By`` de los cuatro fantomas. Minutos en CPU.
- ``svd``: descomposición en valores singulares densa de BH (65 536² elementos).
  Una factorización de ~8.4 h que necesita ~120 GB de RAM y MKL ILP64.
- ``prior``: el modelo de difusión publicado por OpenAI (sin ajuste fino).

La raíz se localiza, en orden: el argumento explícito, ``$DRUS_ARTIFACTS``,
``<repo>/metodos/04_drus/artefactos`` y ese mismo subdirectorio relativo al
directorio de trabajo.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .observations import PHANTOMS, compute_by, load_picmus_rf
from .physics_picmus import build_h_and_b, build_scan_grid
from .prior import PRIOR_FILENAME, PRIOR_MD5, PRIOR_SIZE_BYTES, PRIOR_URL

__all__ = ["GRID_NX", "GRID_NZ", "GRID_X_MM", "GRID_Z_MM", "Artifacts",
           "resolve_artifacts", "build_operator", "stage_bh", "stage_svd",
           "download_prior"]

# Rejilla de la tesis (extendida respecto de la del artículo, z ∈ [10, 46] mm,
# para que entren los 7 alambres del fantoma experimental de resolución).
GRID_NX, GRID_NZ = 256, 256
GRID_X_MM = (-18.0, 18.0)
GRID_Z_MM = (9.0, 48.0)

_SUBPATH = Path("metodos") / "04_drus" / "artefactos"


@dataclass(frozen=True)
class Artifacts:
    """Rutas de los tres artefactos bajo una raíz común."""

    root: Path

    @property
    def operador(self) -> Path:
        return self.root / "operador"

    @property
    def svd(self) -> Path:
        return self.root / "svd"

    @property
    def prior(self) -> Path:
        return self.root / "prior" / PRIOR_FILENAME

    def status(self) -> dict:
        """Qué piezas existen. Útil para ``drus info``."""
        op, sv = self.operador, self.svd
        return {
            "operador": (op / "B.npz").exists() and (op / "meta.json").exists(),
            "svd": all((sv / f).exists() for f in ("S.npy", "Vt.npy")),
            "svd_U": (sv / "U.npy").exists(),
            "prior": self.prior.exists(),
        }

    def missing(self) -> list[str]:
        st = self.status()
        return [k for k in ("operador", "svd", "prior") if not st[k]]


def _candidate_roots() -> list[Path]:
    out: list[Path] = []
    env = os.environ.get("DRUS_ARTIFACTS")
    if env:
        out.append(Path(env).expanduser())
    here = Path(__file__).resolve()
    out.append(here.parents[2] / _SUBPATH)          # <repo>/codigo/drus -> <repo>
    cwd = Path.cwd()
    out += [cwd / _SUBPATH, cwd.parent / _SUBPATH]
    return out


def resolve_artifacts(root: Union[str, Path, None] = None,
                      require: bool = True) -> Artifacts:
    """Localiza la raíz de artefactos. Con ``require`` exige las tres piezas."""
    if root is not None:
        cands = [Path(root).expanduser()]
    else:
        cands = _candidate_roots()
    for c in cands:
        art = Artifacts(c)
        if not require or not art.missing():
            return art
    tried = "\n  ".join(str(c) for c in cands)
    raise FileNotFoundError(
        "No se encontraron los artefactos de DRUS (operador, SVD y prior). "
        "Se buscó en:\n  " + tried + "\n\nSe construyen una sola vez con:\n"
        "  drus prepare operador --picmus <ruta a PICMUS>   # minutos, CPU\n"
        "  drus prepare svd                                 # ~8.4 h, ~120 GB de RAM, MKL\n"
        "  drus prepare prior                               # descarga 2.2 GB\n"
        "y se localizan con DRUS_ARTIFACTS=/ruta o --artifacts /ruta.")


# ─────────────────────────────────────────────── operador (Fase A)
def build_operator(picmus_root: Union[str, Path], out_dir: Union[str, Path],
                   nx: int = GRID_NX, nz: int = GRID_NZ,
                   x_lim_mm: tuple[float, float] = GRID_X_MM,
                   z_lim_mm: tuple[float, float] = GRID_Z_MM) -> dict:
    """Construye H, B y las observaciones ``By`` de los cuatro fantomas.

    Misma cadena que ``metodos/04_drus/scripts/build_by_picmus.py`` (sin las
    figuras de control): la geometría de referencia es la de la simulación de
    resolución, como ``datasetReader(1,1,2)`` en el MATLAB original.
    """
    import scipy.sparse as sp

    picmus_root, out_dir = Path(picmus_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x_axis, z_axis = build_scan_grid(
        nx, nz, x_lim=(x_lim_mm[0] * 1e-3, x_lim_mm[1] * 1e-3),
        z_lim=(z_lim_mm[0] * 1e-3, z_lim_mm[1] * 1e-3))

    ref = load_picmus_rf(picmus_root, "simu_reso")
    t0 = time.time()
    mm = build_h_and_b(ref.probe_x, ref.fs, ref.c, x_axis, z_axis)
    print(f"[operador] H {mm.H.shape} nnz={mm.H.nnz/1e6:.1f}M | B {mm.B.shape} | "
          f"K={mm.K} k_min={mm.k_min} | {time.time()-t0:.1f}s", flush=True)
    sp.save_npz(out_dir / "H.npz", mm.H)
    sp.save_npz(out_dir / "B.npz", mm.B)
    np.savez(out_dir / "grid.npz", x_axis=x_axis, z_axis=z_axis)
    meta = {"K": mm.K, "k_min": mm.k_min, "Ne": len(ref.probe_x),
            "fs": ref.fs, "c": ref.c, "nx": nx, "nz": nz,
            "zlim_mm": list(z_lim_mm), "xlim_mm": list(x_lim_mm),
            "rx_f_number": 1.4, "apod": "tukey25", "ros": 10}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    for phantom in PHANTOMS:
        rf = load_picmus_rf(picmus_root, phantom)
        if abs(rf.fs - ref.fs) >= 1 or abs(rf.c - ref.c) >= 1e-6:
            raise ValueError(f"la geometría de {phantom} difiere de la de referencia")
        by, _ = compute_by(mm.B, rf, mm.K, mm.k_min, phantom)
        np.save(out_dir / f"By_{phantom}.npy", by)
        print(f"[operador] By_{phantom}.npy", flush=True)
    return meta


# ──────────────────────────────────────────────────── SVD (Fase B)
def stage_bh(in_dir: Path, out_dir: Path, block: int = 2048) -> None:
    """BH = B @ H denso por bloques de columnas, escrito a ``BH.npy`` (memmap).

    GPU si la hay, si no CPU; la RAM pico es de ~4 GB.
    """
    import scipy.sparse as sp
    import torch

    in_dir, out_dir = Path(in_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    B = sp.load_npz(in_dir / "B.npz").tocsr()
    H = sp.load_npz(in_dir / "H.npz").tocsc()
    npix = B.shape[0]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[bh] B {B.shape} nnz={B.nnz/1e6:.0f}M | device={dev}", flush=True)

    Bt = torch.sparse_csr_tensor(
        torch.from_numpy(B.indptr.astype(np.int64)),
        torch.from_numpy(B.indices.astype(np.int64)),
        torch.from_numpy(B.data),
        size=B.shape, dtype=torch.float32, device=dev)

    out = np.lib.format.open_memmap(out_dir / "BH.npy", mode="w+",
                                    dtype=np.float32, shape=(npix, npix))
    for c0 in range(0, npix, block):
        c1 = min(c0 + block, npix)
        hblk = torch.from_numpy(H[:, c0:c1].toarray()).to(dev)
        out[:, c0:c1] = (Bt @ hblk).cpu().numpy()
        if (c0 // block) % 8 == 0:
            print(f"[bh] cols {c1}/{npix}  {time.time()-t0:.0f}s", flush=True)
    out.flush()
    del out
    print(f"[bh] BH.npy escrito ({npix}x{npix} f32) en {time.time()-t0:.0f}s", flush=True)


def _mem_available_gib() -> float:
    """Mínimo entre MemAvailable del nodo y el límite cgroup del job (menos uso actual)."""
    avail = 0.0
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable"):
                avail = int(line.split()[1]) / 2**20
                break
    for lim_f, use_f in [("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current"),
                         ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
                          "/sys/fs/cgroup/memory/memory.usage_in_bytes")]:
        try:
            lim = Path(lim_f).read_text().strip()
            if lim not in ("max", ""):
                cg = (int(lim) - int(Path(use_f).read_text())) / 2**30
                if 0 < cg < 4096:
                    avail = min(avail, cg)
            break
        except (FileNotFoundError, ValueError, PermissionError):
            continue
    return avail


def _svd_gram(bh: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fallback: SVD vía eigendescomposición de la Gram G = BHᵀ·BH (ssyevd).

    σ = sqrt(λ) pierde precisión relativa bajo ~sqrt(eps_f32)·σmax; aceptable
    para DDRM (los σ pequeños solo participan en comparaciones de umbral).
    """
    from .svd_bh import load_mkl_ilp64, ssyevd_ilp64

    n = bh.shape[0]
    mkl = load_mkl_ilp64()
    print("[svd] ruta GRAM: G = BHt @ BH", flush=True)
    g = bh.T @ bh                                    # 16 GiB extra
    lam, ev = ssyevd_ilp64(g, mkl=mkl)               # ev FILAS = eigenvectores, λ ascendente
    vt = ev[::-1, :].copy()                          # Vt: fila k = vector singular derecho k (desc)
    del g, ev
    s = np.sqrt(np.clip(lam[::-1], 0.0, None)).astype(np.float32)
    # U = BH V S⁻¹ (bloqueado); columnas con σ≈0 quedan escaladas a ~0 (no
    # ortonormales ahí — documentado; DDRM no las usa numéricamente).
    u = np.empty((n, n), dtype=np.float32)
    s_safe = np.where(s > s[0] * 1e-7, s, np.float32(np.inf))
    blk = 4096
    for c0 in range(0, n, blk):
        c1 = min(c0 + blk, n)
        u[:, c0:c1] = (bh @ vt[c0:c1, :].T) / s_safe[c0:c1][None, :]
    return u, s, vt


def stage_svd(in_dir: Path, out_dir: Path, n_check: int = 64) -> dict:
    """SVD completa de ``BH.npy`` y ``Uᵀ·By`` de las observaciones de ``in_dir``.

    sgesdd MKL ILP64 (~6·n² de memoria) con fallback automático a la ruta Gram
    si la RAM disponible no alcanza. Verifica la factorización sobre columnas
    muestreadas (gate G-B1) y guarda ``S.npy``, ``U.npy``, ``Vt.npy``,
    ``Uty_<fantoma>.npy``, ``spectrum.png`` y ``svd_meta.json``.
    """
    from .svd_bh import load_mkl_ilp64, sgesdd_ilp64

    in_dir, out_dir = Path(in_dir), Path(out_dir)
    t0 = time.time()
    bh = np.load(out_dir / "BH.npy")                # 16 GiB a RAM
    n = bh.shape[0]
    rng = np.random.default_rng(0)
    jcols = np.sort(rng.choice(n, size=min(n_check, n), replace=False))
    bh_sample = bh[:, jcols].copy()

    avail = _mem_available_gib()
    n2_gib = n * n * 4 / 2**30
    # gesdd necesita además de BH (ya residente): U (n²) + VT (n²) + work (~4n²)
    need_gesdd = 6 * n2_gib
    method = "gesdd" if avail > need_gesdd + 4 else "gram"
    print(f"[svd] MemAvailable={avail:.0f} GiB | n²={n2_gib:.0f} GiB | "
          f"necesita~{need_gesdd:.0f} GiB extra → método={method} | "
          f"t={time.time()-t0:.0f}s", flush=True)

    if method == "gesdd":
        try:
            u, s, vt = sgesdd_ilp64(bh, mkl=load_mkl_ilp64())
        except MemoryError:
            print("[svd] MemoryError en gesdd → fallback gram", flush=True)
            bh = np.load(out_dir / "BH.npy")
            u, s, vt = _svd_gram(bh)
    else:
        u, s, vt = _svd_gram(bh)
    del bh
    print(f"[svd] SVD lista en {(time.time()-t0)/60:.1f} min | "
          f"σmax={s[0]:.4e} σmin={s[-1]:.4e}", flush=True)

    # --- Gate G-B1: reconstrucción en columnas muestreadas ---
    recon = u @ (s[:, None] * vt[:, jcols])
    rel = float(np.linalg.norm(recon - bh_sample) / np.linalg.norm(bh_sample))
    k = min(2000, n)
    ortho = float(np.abs(vt[:k] @ vt[:k].T - np.eye(k)).max())
    print(f"[G-B1] rel_err({len(jcols)} cols) = {rel:.3e} | max|VtVᵀ−I| = {ortho:.2e} | "
          f"NaN en S: {np.isnan(s).any()}", flush=True)

    np.save(out_dir / "S.npy", s)
    np.save(out_dir / "U.npy", u)
    np.save(out_dir / "Vt.npy", vt)

    # --- U^T · By de los fantomas (lo único que el muestreador necesita de U) ---
    done = []
    for f in sorted(in_dir.glob("By_*.npy")):
        by = np.load(f).astype(np.float32)
        np.save(out_dir / f"Uty_{f.stem[3:]}.npy", u.T @ by)
        done.append(f.stem[3:])
    print(f"[svd] Uty guardado para: {done}", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(np.maximum(s, 1e-12))
    ax.set_xlabel("índice"); ax.set_ylabel("σ")
    ax.set_title(f"Espectro singular BH (G-B1 rel={rel:.1e})")
    fig.savefig(out_dir / "spectrum.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    meta = {"n": int(n), "method": method, "gate_b1_rel_err": rel,
            "vt_ortho_2k": ortho, "sigma_max": float(s[0]), "sigma_min": float(s[-1]),
            "minutes": (time.time() - t0) / 60,
            "convention": "BH = U @ diag(S) @ Vt (numpy C-order)"}
    (out_dir / "svd_meta.json").write_text(json.dumps(meta, indent=2))
    print("[svd] completa", flush=True)
    return meta


# ───────────────────────────────────────────────────────────────── prior
def _md5(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def download_prior(dst: Union[str, Path], url: str = PRIOR_URL,
                   verify: bool = True) -> Path:
    """Descarga el prior publicado por OpenAI y comprueba tamaño y MD5."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        tmp = dst.with_suffix(dst.suffix + ".part")
        print(f"[prior] descargando {url}\n        -> {dst}", flush=True)
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
            done = 0
            while True:
                blk = r.read(1 << 24)
                if not blk:
                    break
                f.write(blk)
                done += len(blk)
                print(f"\r[prior] {done/2**30:5.2f} / {PRIOR_SIZE_BYTES/2**30:.2f} GiB",
                      end="", flush=True)
        print()
        tmp.rename(dst)
    if verify:
        size = dst.stat().st_size
        if size != PRIOR_SIZE_BYTES:
            raise ValueError(f"{dst}: {size} bytes, se esperaban {PRIOR_SIZE_BYTES}")
        md5 = _md5(dst)
        if md5 != PRIOR_MD5:
            raise ValueError(f"{dst}: MD5 {md5} != {PRIOR_MD5}")
        print(f"[prior] verificado (MD5 {md5})", flush=True)
    return dst


def picmus_root_from(path: Optional[Union[str, Path]]) -> Path:
    """Raíz de PICMUS: el argumento, o ``$TESIS_DATASETS/PICMUS``."""
    if path is not None:
        return Path(path).expanduser()
    base = os.environ.get("TESIS_DATASETS")
    if not base:
        raise FileNotFoundError("Indica la raíz de PICMUS (--picmus) o define "
                                "TESIS_DATASETS con un subdirectorio PICMUS/.")
    return Path(base) / "PICMUS"

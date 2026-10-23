"""Modelo físico DRUS para PICMUS: H (forward) y B (beamformer adjunto apodizado).

Port exacto de ``MATLAB/Results/02_picmus/BH/compute_svdBH.m`` del repo DRUS-v1
(Zhang et al. 2023). Diferencias deliberadas respecto al MATLAB, documentadas:

1. **Ordenamiento de píxeles**: MATLAB usa column-major con z variando más
   rápido y luego aplica una permutación a V para el sampler Python. Aquí el
   vector imagen usa directamente la convención row-major de numpy/torch
   ``p = iz * Nx + ix`` (una fila del grid por cada z), de modo que
   ``x.reshape(Nz, Nx)`` produce la imagen sin permutación alguna.
2. **Sin truco de espejo L/R**: MATLAB construye media matriz y la refleja por
   eficiencia. Aquí se construye el grid completo vectorizado por elemento;
   el resultado es el mismo operador.

Todo lo demás replica el MATLAB operación por operación, incluyendo el banco
de pulso con stride ``ros+1`` sobre un pulso muestreado a ``dth = 1/(fs*ros)``
(quirk del original: el banco re-muestrea efectivamente el pulso a fs/1.1;
se replica tal cual porque así se generaron los resultados del paper).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.interpolate import CubicSpline
from scipy.signal import fftconvolve


# ---------------------------------------------------------------------------
# Grid del paper: 256×256, x ∈ [−18, 18] mm, z ∈ [10, 46] mm
# ---------------------------------------------------------------------------

def build_scan_grid(nx: int = 256, nz: int = 256,
                    x_lim: tuple[float, float] = (-0.018, 0.018),
                    z_lim: tuple[float, float] = (0.01, 0.046)):
    """Grid del paper por default. Variante extendida z_lim=(0.009, 0.048)
    cubre los 7 wires de expe_reso (z∈[9.6, 47.0] mm) para el eval n_lat/7
    de unet_ag."""
    x_axis = np.linspace(x_lim[0], x_lim[1], nx)
    z_axis = np.linspace(z_lim[0], z_lim[1], nz)
    return x_axis, z_axis


def _matlab_colon(lo: float, step: float, hi: float) -> np.ndarray:
    """Réplica de ``lo:step:hi`` de MATLAB (incluye hi si cae en la malla)."""
    n = int(math.floor((hi - lo) / step + 1e-10)) + 1
    return lo + np.arange(n) * step


# ---------------------------------------------------------------------------
# Pulso PICMUS (3 convoluciones gaussianas) → banco de delays fraccionales
# ---------------------------------------------------------------------------

@dataclass
class PulseBank:
    bank: np.ndarray   # (Kh, ros+1) — columna j = pulso a fracción de delay j/ros
    lagt: float        # lag temporal del pulso [s]
    kh: int
    ros: int


def make_pulse_bank(fs: float, ros: int = 10) -> PulseBank:
    dth = 1.0 / fs / ros

    fe, bwre, ncycles = 5.208e6, 0.67, 2.5
    ae = 4 * math.log(2) / (math.pi * fe * bwre) ** 2
    te = _matlab_colon(-ncycles / fe / 2, dth, ncycles / fe / 2)
    he = np.exp(-te ** 2 / ae) * np.cos(2 * math.pi * fe * te)

    f0, bwr0 = 5.1333e6, 0.65
    a0 = 4 * math.log(2) / (math.pi * f0 * bwr0) ** 2
    t0 = _matlab_colon(-2 / f0, dth, 2 / f0)
    h0 = np.exp(-t0 ** 2 / a0) * np.cos(2 * math.pi * f0 * t0)

    one_way = fftconvolve(h0, he)
    h = fftconvolve(one_way, h0)
    h = h / np.max(np.abs(h))  # MATLAB: h./max(h(:)); max>0 aquí (pico central positivo)

    nth = len(h)
    # MATLAB h(fix(Nth/4) : ceil(Nth/4*3)) — 1-based inclusivo
    h = h[int(nth / 4) - 1 : math.ceil(nth / 4 * 3)]
    nth = len(h)
    lagt = (nth / 2) * dth

    kh = nth // (ros + 1)
    # MATLAB: reshape(h(1:Kh*(ros+1)), ros+1, Kh)'  ≡  numpy reshape(Kh, ros+1)
    bank = h[: kh * (ros + 1)].reshape(kh, ros + 1)
    bank = bank[:, ::-1].copy()  # flip(h, 2)
    return PulseBank(bank=bank, lagt=lagt, kh=kh, ros=ros)


# ---------------------------------------------------------------------------
# Apodización RX (réplica de tools.apodization, ventana tukey25)
# ---------------------------------------------------------------------------

def tukey25(distance: np.ndarray, aperture: np.ndarray) -> np.ndarray:
    roll = 0.25
    d, a = distance, aperture
    flat = (d < (a / 2 * (1 - roll))).astype(np.float64)
    taper = ((d > (a / 2 * (1 - roll))) & (d < (a / 2))) * 0.5 * (
        1 + np.cos(2 * math.pi / roll * (d / a - roll / 2 - 0.5))
    )
    return flat + taper


# ---------------------------------------------------------------------------
# Construcción de H y B
# ---------------------------------------------------------------------------

@dataclass
class ModelMatrices:
    H: sp.csr_matrix          # (K*Ne, Npix)
    B: sp.csr_matrix          # (Npix, K*Ne)
    K: int                    # samples temporales por elemento
    k_min: int                # primer sample con señal (1-based, como MATLAB)
    x_axis: np.ndarray
    z_axis: np.ndarray


def build_h_and_b(
    probe_x: np.ndarray,      # (Ne,) posiciones x de los elementos [m]
    fs: float,
    c: float,
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    rx_f_number: float = 1.4,
    ros: int = 10,
    dtype=np.float32,
    pulse: PulseBank | None = None,
) -> ModelMatrices:
    ne = len(probe_x)
    nx, nz = len(x_axis), len(z_axis)
    npix = nx * nz

    if pulse is None:
        pulse = make_pulse_bank(fs, ros)
    bank, kh, lagt = pulse.bank, pulse.kh, pulse.lagt

    # Vectores de píxeles en convención row-major p = iz*Nx + ix
    x = np.tile(x_axis, nz)              # (Npix,)
    z = np.repeat(z_axis, nx)            # (Npix,)

    # --- Delay TX (plane wave 0°): min distancia a la apertura interpolada ---
    # MATLAB: interp1(xe, linspace(1,Ne,4Ne), 'spline') — evalúa xe en índices
    # fraccionales (spline de datos lineales ≡ lineal, se replica igual).
    idxi = np.linspace(1, ne, 4 * ne)
    xti = CubicSpline(np.arange(1, ne + 1), probe_x)(idxi)      # (4Ne,)
    # dTX por chunks para no materializar (Npix, 4Ne) completo en float64
    dtx = np.empty(npix)
    chunk = 8192
    for s in range(0, npix, chunk):
        e = min(s + chunk, npix)
        dtx[s:e] = np.sqrt((xti[None, :] - x[s:e, None]) ** 2 + z[s:e, None] ** 2).min(axis=1)

    drx = np.sqrt((probe_x[None, :] - x[:, None]) ** 2 + z[:, None] ** 2)   # (Npix, Ne)

    tau = (dtx[:, None] + drx) / c - lagt                                   # (Npix, Ne)
    delay = tau * fs + 1.0                                                  # 1-based, como MATLAB

    k_min = int(np.floor(delay.min()))
    big_k = int(np.ceil(delay.max())) + kh

    # --- Apodización RX ---
    rx_aperture = z / rx_f_number                                           # (Npix,)
    apod = tukey25(np.abs(x[:, None] - probe_x[None, :]), rx_aperture[:, None])  # (Npix, Ne)

    # --- Ensamblado sparse por elemento ---
    di = np.floor(delay).astype(np.int64)          # parte entera (1-based)
    dd = delay - di                                # fracción ∈ [0,1)
    frac_col = np.rint(dd * ros).astype(np.int64)  # columna del banco, 0..ros

    pix_idx = np.arange(npix, dtype=np.int64)
    rows_parts, cols_parts, hvals_parts, bvals_parts = [], [], [], []
    for e in range(ne):
        base = di[:, e] - 1 + e * big_k                          # fila 0-based del primer sample
        rows = (base[:, None] + np.arange(kh)[None, :]).ravel()  # (Npix*Kh,)
        cols = np.repeat(pix_idx, kh)
        vals = bank[:, frac_col[:, e]].T.ravel()                 # (Npix, Kh) → flat pixel-major
        rows_parts.append(rows.astype(np.int64))
        cols_parts.append(cols)
        hvals_parts.append(vals.astype(dtype))
        bvals_parts.append((vals * np.repeat(apod[:, e], kh)).astype(dtype))

    rows = np.concatenate(rows_parts)
    cols = np.concatenate(cols_parts)
    hvals = np.concatenate(hvals_parts)
    bvals = np.concatenate(bvals_parts)
    del rows_parts, cols_parts, hvals_parts, bvals_parts

    shape_h = (big_k * ne, npix)
    H = sp.coo_matrix((hvals, (rows, cols)), shape=shape_h).tocsr()
    B = sp.coo_matrix((bvals, (cols, rows)), shape=(npix, big_k * ne)).tocsr()

    return ModelMatrices(H=H, B=B, K=big_k, k_min=k_min, x_axis=x_axis, z_axis=z_axis)

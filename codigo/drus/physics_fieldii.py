"""Modelo físico DRUS para la geometría Field II multiangle (Fase F.1).

Reusa ``build_h_and_b`` de ``physics_picmus`` (misma sonda: 128 el., pitch
300 µm, fs 20.832 MHz) con dos adaptaciones documentadas:

1. **Banco de pulso Field II**: el dataset se generó con
   ``generate_training_data_multiangle.m``:
       excitación  : sin(2π·fc·t), 2.5 ciclos @ fc=5.208 MHz (burst causal)
       IR (tx=rx)  : sin(2π·fc·t)·exp(−π²·(0.67·fc)²·t²), t ∈ [−2/fc, 2/fc]
   El two-way h = ir ⊛ exc ⊛ ir se construye aquí analíticamente a paso
   dth = 1/(fs·ros), centrado, y se recorta a la mitad central con el mismo
   procedimiento (y convención de ``lagt``) que ``make_pulse_bank`` para que
   la semántica de delays de ``build_h_and_b`` no cambie. El BW resultante
   (~1.8 MHz) difiere del banco PICMUS (~1.0 MHz) — usar el banco PICMUS
   aquí sería un model mismatch espectral del −43 %.

2. **Alineación temporal**: Field II devuelve ``t_start`` por emisión
   (ángulo 0°: ~3.46 µs ≈ 72.0 samples). H asume y indexada desde t=0, así
   que la observación se antepone con ``round(t_start·fs)`` ceros más una
   corrección global ``pad_extra`` calibrada empíricamente contra las
   posiciones GT de los wires (gate del build script); el residuo de la
   convención de lag del pulso queda absorbido ahí.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import fftconvolve

from .physics_picmus import PulseBank, _matlab_colon


def make_pulse_bank_fieldii(
    fs: float,
    ros: int = 10,
    fc: float = 5.208e6,
    ncycles: float = 2.5,
    frac_bw: float = 0.67,
) -> PulseBank:
    """Banco de pulso two-way del generador Field II, misma convención que
    ``make_pulse_bank`` (crop mitad central, lagt = mitad del pulso recortado)."""
    dth = 1.0 / fs / ros

    # Excitación: burst causal de 2.5 ciclos, centrado para la convención lagt.
    te = _matlab_colon(-ncycles / fc / 2, dth, ncycles / fc / 2)
    he = np.sin(2 * math.pi * fc * (te + ncycles / fc / 2))

    # IR del transductor (idéntica en TX y RX).
    t0 = _matlab_colon(-2 / fc, dth, 2 / fc)
    h0 = np.sin(2 * math.pi * fc * t0) * np.exp(-(math.pi * frac_bw * fc) ** 2 * t0 ** 2)

    h = fftconvolve(fftconvolve(h0, he), h0)
    h = h / np.max(np.abs(h))

    nth = len(h)
    h = h[int(nth / 4) - 1: math.ceil(nth / 4 * 3)]
    nth = len(h)
    lagt = (nth / 2) * dth

    kh = nth // (ros + 1)
    bank = h[: kh * (ros + 1)].reshape(kh, ros + 1)
    bank = bank[:, ::-1].copy()
    return PulseBank(bank=bank, lagt=lagt, kh=kh, ros=ros)


def align_fieldii_rf(
    rf_0deg: np.ndarray,      # (D, Ne) RF del ángulo 0°
    t_start: float,           # t_start del ángulo 0° [s]
    fs: float,
    big_k: int,
    pad_extra: int = 0,
) -> np.ndarray:
    """Antepone ceros para llevar la RF a la convención t=0 de H y recorta/rellena
    a exactamente ``big_k`` samples. ``pad_extra`` es la corrección calibrada."""
    n0 = int(round(t_start * fs)) + int(pad_extra)
    d, ne = rf_0deg.shape
    y = np.zeros((big_k, ne), dtype=np.float64)
    if n0 >= 0:
        n_copy = min(d, big_k - n0)
        if n_copy > 0:
            y[n0:n0 + n_copy, :] = rf_0deg[:n_copy, :]
    else:
        n_copy = min(d + n0, big_k)
        if n_copy > 0:
            y[:n_copy, :] = rf_0deg[-n0:-n0 + n_copy, :]
    return y


def compute_by_fieldii(
    B,                        # sp.csr_matrix (Npix, K*Ne)
    rf_0deg: np.ndarray,      # (D, Ne)
    t_start: float,
    fs: float,
    big_k: int,
    k_min: int,
    pad_extra: int = 0,
    tgc_alpha: float = 4.0,
    norm_scale: float = 10.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Espejo de ``observations.compute_by`` para RF Field II: alineación
    t_start + TGC exponencial con clamp near-field + normalización."""
    y = align_fieldii_rf(rf_0deg, t_start, fs, big_k, pad_extra)

    t = np.arange(big_k) / fs
    tgc = np.exp(tgc_alpha * t / t.max())
    tgc[: k_min + 150] = tgc[k_min + 150]
    y *= tgc[:, None]

    y /= np.max(np.abs(y)) + 1e-30
    y *= norm_scale

    by = B @ y.flatten(order="F")
    return by.astype(np.float64), y

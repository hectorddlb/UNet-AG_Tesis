"""Sampler DDRM para DRUS — Fase D.

Port de ``functions/denoising.py::efficient_generalized_steps`` y
``functions/svd_replacement.py::ultrasound0`` del repo DRUS-v1, adaptado a
nuestros artefactos de Fase B:

- SVD en convención numpy C-order: ``BH = U @ diag(S) @ Vt`` con el vector
  imagen row-major ``p = iz*Nx + ix`` (sin las permutaciones que el original
  necesitaba para MATLAB column-major).
- ``Uty = Uᵀ·By`` viene precomputado del job de Fase B (el sampler nunca
  necesita U completa: y es fijo por phantom).

Layout de vectores (idéntico al original para replicar la aritmética):
- espacio imagen: flat channel-major ``[c0 p0..pN, c1 p0..pN, c2 ...]``
  (= ``x.view(B, -1)`` de un tensor (B,3,S,S)).
- espacio espectral: flat pixel-major con tripletes ``[m0c0 m0c1 m0c2, m1c0...]``
  (modo espectral × canal), con σ_k replicado 3 veces consecutivas.

El modelo (UNet 553M, entrada RGB) corre en GPU; los matmuls espectrales con
Vt (17 GB) corren en CPU — igual que el código original de Zhang. γ se
multiplica por √3 como en el original (y replicado a 3 canales).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

CHANNELS = 3


# ---------------------------------------------------------------------------
# Operador espectral (reemplazo de ultrasound0)
# ---------------------------------------------------------------------------

class SpectralOps:
    """Multiplicaciones por V/Vᵀ replicadas a 3 canales.

    ``vt``: (N, N) float32 — filas = vectores singulares derechos (numpy Vt).
    Los productos corren en ``spectral_device`` (CPU por tamaño).
    """

    def __init__(self, s: np.ndarray, vt: np.ndarray, spectral_device: str = "cpu"):
        self.dev = torch.device(spectral_device)
        self.vt = torch.from_numpy(vt).to(self.dev)            # (N, N)
        self.n = self.vt.shape[0]
        s_t = torch.from_numpy(np.ascontiguousarray(s)).float()
        self.singulars = s_t.repeat_interleave(CHANNELS).to(self.dev)   # (3N,) tripletes

    # --- imagen (channel-major flat) → espectral (pixel-major flat) ---
    def Vt(self, vec: torch.Tensor) -> torch.Tensor:
        b = vec.shape[0]
        x = vec.reshape(b, CHANNELS, self.n).permute(0, 2, 1).to(self.dev)  # (B,N,3)
        out = torch.matmul(self.vt, x)                                       # (B,N,3)
        return out.reshape(b, -1)                                            # pixel-major

    # --- espectral (pixel-major flat) → imagen (channel-major flat) ---
    def V(self, vec: torch.Tensor) -> torch.Tensor:
        b = vec.shape[0]
        x = vec.reshape(b, self.n, CHANNELS).to(self.dev)                    # (B,N,3)
        out = torch.matmul(self.vt.transpose(0, 1), x)                       # (B,N,3)
        return out.permute(0, 2, 1).reshape(b, -1)                           # channel-major


def expand_uty(uty: np.ndarray) -> torch.Tensor:
    """Uᵀ·By (N,) → (1, 3N) pixel-major (equivale a Ut(y repetido a 3 canales))."""
    t = torch.from_numpy(np.ascontiguousarray(uty)).float()
    return t.repeat_interleave(CHANNELS)[None, :]


# ---------------------------------------------------------------------------
# DDRM (efficient_generalized_steps)
# ---------------------------------------------------------------------------

@dataclass
class DDRMResult:
    x_final: np.ndarray     # (3, S, S) — x_t final (≡ xs[-1] del original: lo que evalúa el paper)
    x_mean: np.ndarray      # (S, S) — promedio de canales de x_final
    x0_pred: np.ndarray     # (3, S, S) — última predicción x0 del modelo (diagnóstico)
    seconds: float
    gamma_effective: float


def _compute_alpha(beta: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    beta = torch.cat([torch.zeros(1, device=beta.device), beta], dim=0)
    return (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)


@torch.no_grad()
def ddrm_sample(
    model: torch.nn.Module,
    betas: np.ndarray,           # (1000,) linear 1e-4..0.02 — mismas del FT
    ops: SpectralOps,
    uty: torch.Tensor,           # (1, 3N) de expand_uty
    gamma: float,                # σ del ruido de observación (HP, grid del paper)
    image_size: int = 256,
    timesteps: int = 50,
    eta: float = 0.85,           # etaA = etaC (defaults del repo DRUS-v1)
    eta_b: float = 1.0,
    seed: int = 0,
    model_device: str = "cuda",
    verbose: bool = True,
) -> DDRMResult:
    t_start = time.time()
    dev_m = torch.device(model_device)
    dev_s = ops.dev
    b = torch.from_numpy(betas.astype(np.float32)).to(dev_s)
    num_t = b.shape[0]

    sigma_0 = float(gamma) * np.sqrt(CHANNELS)     # como el original (in_channels==3)
    singulars = ops.singulars                       # (3N,)
    u_t_y = uty.to(dev_s)                           # (1, 3N)
    sig_inv_u_t_y = u_t_y / singulars.clamp_min(1e-30)[None, :]

    g = torch.Generator(device="cpu").manual_seed(seed)
    shape = (1, CHANNELS, image_size, image_size)
    x = torch.randn(*shape, generator=g)

    skip = num_t // timesteps
    seq = list(range(0, num_t, skip))
    seq_next = [-1] + seq[:-1]

    # ---- init x_T | y (p(x_T | x0, y) del paper DDRM) ----
    largest_alpha = _compute_alpha(b, torch.tensor([seq[-1]], device=dev_s).long())
    largest_sigma = ((1 - largest_alpha).sqrt() / largest_alpha.sqrt()).item()
    large_idx = singulars * largest_sigma > sigma_0
    inv_sing_zero = torch.zeros_like(singulars)
    inv_sing_zero[large_idx] = sigma_0 / singulars[large_idx]

    init_y = torch.zeros(1, CHANNELS * ops.n, device=dev_s)
    init_y[0, large_idx] = u_t_y[0, large_idx] / singulars[large_idx]
    remaining = (largest_sigma ** 2 - inv_sing_zero ** 2).clamp_min(0.0).sqrt()
    init_y = init_y + remaining[None, :] * x.view(1, -1).to(dev_s)
    init_y = init_y / largest_sigma
    xt = ops.V(init_y).view(*shape)                # imagen channel-major

    x0_t = None
    for i, j in zip(reversed(seq), reversed(seq_next)):
        t = torch.tensor([i], device=dev_s).long()
        t_next = torch.tensor([j], device=dev_s).long()
        at = _compute_alpha(b, t)
        at_next = _compute_alpha(b, t_next)
        sigma = ((1 - at).sqrt() / at.sqrt()).item()
        sigma_next = ((1 - at_next).sqrt() / at_next.sqrt()).item()

        # --- UNet en GPU ---
        et = model(xt.to(dev_m), torch.tensor([i], device=dev_m).float())
        if et.size(1) == 2 * CHANNELS:
            et = et[:, :CHANNELS]                   # learn_sigma: descarta varianza
        et = et.to(dev_s).float()

        x0_t = (xt.to(dev_s) - et * (1 - at).sqrt().item()) / at.sqrt().item()

        # --- actualización espectral (3 casos DDRM) ---
        v_t_x0 = ops.Vt(x0_t.view(1, -1))                       # (1, 3N)
        sv_t_x0 = v_t_x0 * singulars[None, :]

        cond_before = singulars * sigma_next > sigma_0
        cond_after = singulars * sigma_next < sigma_0

        std_c = sigma_next * eta
        sig_tilde_c = float(np.sqrt(max(sigma_next ** 2 - std_c ** 2, 0.0)))
        std_a = sigma_next * eta
        sig_tilde_a = float(np.sqrt(max(sigma_next ** 2 - std_a ** 2, 0.0)))

        vt_et = ops.Vt(et.view(1, -1))
        noise = torch.randn(v_t_x0.shape, generator=g).to(dev_s)

        # caso "missing" (σ = 0 no ocurre aquí, pero mantiene la forma del original)
        vt_next = v_t_x0 + sig_tilde_c * vt_et + std_c * noise

        # caso "after": σ·σ_next < σ_0 — jala hacia y con peso relativo
        vt_next[:, cond_after] = (
            v_t_x0[:, cond_after]
            + sig_tilde_a * ((u_t_y - sv_t_x0) / sigma_0)[:, cond_after]
            + std_a * noise[:, cond_after]
        )

        # caso "before": σ·σ_next > σ_0 — mezcla con S⁻¹Uᵀy
        if cond_before.any():
            diff_sig = (sigma_next ** 2
                        - (sigma_0 / singulars[cond_before]) ** 2 * eta_b ** 2).clamp_min(0.0).sqrt()
            noise_b = torch.randn(u_t_y.shape, generator=g).to(dev_s)
            vt_next[:, cond_before] = (
                sig_inv_u_t_y[:, cond_before] * eta_b
                + (1 - eta_b) * v_t_x0[:, cond_before]
                + diff_sig[None, :] * noise_b[:, cond_before]
            )

        xt = (at_next.sqrt().item() * ops.V(vt_next)).view(*shape)
        if verbose:
            print(f"  [ddrm] t={i:4d} σ={sigma:8.3f} | {time.time()-t_start:6.1f}s", flush=True)

    x_final = xt.view(CHANNELS, image_size, image_size).cpu().numpy().astype(np.float32)
    x0_pred = x0_t.view(CHANNELS, image_size, image_size).cpu().numpy().astype(np.float32)
    return DDRMResult(
        x_final=x_final,
        x_mean=x_final.mean(axis=0),
        x0_pred=x0_pred,
        seconds=time.time() - t_start,
        gamma_effective=sigma_0,
    )

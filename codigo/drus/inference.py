# -*- coding: utf-8 -*-
"""API pública de inferencia de DRUS, el método de difusión comparado.

Encapsula la cadena completa de la réplica —observación ``By`` con el modelo
físico de PICMUS, proyección espectral ``Uᵀ·By``, muestreo DDRM de 50 pasos
con el prior de difusión— detrás de la misma clase de interfaz que el método
propuesto (``unet_ag.inference``).

Uso mínimo
----------
>>> from drus.inference import DRUS
>>> modelo = DRUS.from_artifacts()            # localiza $DRUS_ARTIFACTS
>>> res = modelo.reconstruct_picmus("contrast_speckle_expe_dataset_rf.hdf5")
>>> res.bmode.shape                           # envolvente log [0, 1], 256 x 256
(256, 256)

Condiciones de la tesis (Capítulo 6): prior sin ajuste fino, parámetro
``γ`` elegido por fantoma mediante búsqueda en rejilla —como en el artículo
original— y una semilla de muestreo (0). Sin ``gamma`` explícito,
``reconstruct_picmus`` aplica el ``γ`` de la tesis para el fantoma.

Requisitos: GPU para la UNet de 552.8 M parámetros y ~20 GB de RAM para la
matriz ``Vᵀ`` de la descomposición espectral, cuyos productos corren en CPU
como en el código original. Los artefactos se construyen una vez con
``drus prepare`` (ver ``drus.preparacion``).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import scipy.sparse as sp
import torch

from .ddrm_sampler import SpectralOps, ddrm_sample, expand_uty
from .observations import (PHANTOMS, PicmusRF, compute_by, load_picmus_rf_file,
                           phantom_from_filename)
from .preparacion import Artifacts, resolve_artifacts
from .prior import load_prior, n_parameters

__all__ = ["DRUS", "DRUSResult", "THESIS_GAMMA", "DYN_RANGE_DB", "env_log01",
           "to_grid"]

DYN_RANGE_DB = 60.0
"""Rango dinámico con el que se comprime la envolvente, en decibeles."""

THESIS_GAMMA = {"simu_reso": 75.0, "simu_cont": 50.0,
                "expe_reso": 75.0, "expe_cont": 25.0}
"""``γ`` por fantoma con el que se reportan las métricas de DRUS en la tesis
(búsqueda en la rejilla {20, 25, 50, 75}, semilla 0, prior sin ajuste fino)."""

BETAS = np.linspace(1e-4, 0.02, 1000)
"""Calendario lineal de ruido del prior (1000 pasos)."""


def env_log01(x: np.ndarray, dyn_range_db: float = DYN_RANGE_DB) -> np.ndarray:
    """|x| normalizado a su máximo, en dB recortado a ``-dyn_range_db``, en [0, 1].

    Es la conversión con la que la evaluación de la tesis lleva la salida de
    DRUS (reflectividad con signo) al dominio común ``env_log``.
    """
    env = np.abs(np.asarray(x)).astype(np.float64)
    env = env / (env.max() + 1e-30)
    db = np.clip(20.0 * np.log10(env + 1e-30), -dyn_range_db, 0.0)
    return ((db + dyn_range_db) / dyn_range_db).astype(np.float32)


def to_grid(img: np.ndarray, src_z: np.ndarray, src_x: np.ndarray,
            dst_z: np.ndarray, dst_x: np.ndarray) -> np.ndarray:
    """Interpolación bilineal a otra rejilla, con cero fuera del campo de visión.

    Es la adaptación con la que la tesis evalúa DRUS sobre la rejilla común de
    612 x 388 puntos.
    """
    from scipy.interpolate import RegularGridInterpolator

    f = RegularGridInterpolator((src_z, src_x), np.asarray(img, dtype=np.float64),
                                bounds_error=False, fill_value=0.0)
    zz, xx = np.meshgrid(dst_z, dst_x, indexing="ij")
    return f(np.stack([zz.ravel(), xx.ravel()], axis=-1)) \
        .reshape(len(dst_z), len(dst_x)).astype(np.float32)


@dataclass
class DRUSResult:
    """Salida de una reconstrucción."""

    x: np.ndarray
    """``(256, 256)`` float32: reflectividad con signo en el dominio del prior
    (promedio de los tres canales del último paso DDRM)."""
    bmode: np.ndarray
    """``(256, 256)`` float32 en [0, 1]: ``env_log01(x)``."""
    z_axis: np.ndarray
    x_axis: np.ndarray
    phantom: Optional[str]
    gamma: float
    seed: int
    seconds: float


class DRUS:
    """Reconstructor DRUS (DDRM con el prior de difusión) desde una onda plana.

    Parameters
    ----------
    artifacts : raíz de artefactos (``Artifacts``, ruta o ``None`` para buscarla).
    device : dispositivo de la UNet: ``"cuda"``, ``"cpu"`` o ``"auto"``.
    timesteps, eta, eta_b : parámetros del muestreador (los de DRUS-v1).
    """

    def __init__(self, artifacts: Union[Artifacts, str, Path, None] = None,
                 device: str = "auto", timesteps: int = 50, eta: float = 0.85,
                 eta_b: float = 1.0):
        self.artifacts = artifacts if isinstance(artifacts, Artifacts) \
            else resolve_artifacts(artifacts)
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") \
            if device == "auto" else device
        self.timesteps, self.eta, self.eta_b = int(timesteps), float(eta), float(eta_b)

        op = self.artifacts.operador
        self.meta = json.loads((op / "meta.json").read_text())
        grid = np.load(op / "grid.npz")
        self.x_axis = grid["x_axis"].astype(np.float64)
        self.z_axis = grid["z_axis"].astype(np.float64)
        self._B: Optional[sp.csr_matrix] = None
        self._ops: Optional[SpectralOps] = None
        self._model: Optional[torch.nn.Module] = None

    # ── constructores ────────────────────────────────────────────────────
    @classmethod
    def from_artifacts(cls, root: Union[str, Path, None] = None, **kwargs) -> "DRUS":
        """Carga la réplica desde la raíz de artefactos (``$DRUS_ARTIFACTS``)."""
        return cls(resolve_artifacts(root), **kwargs)

    # ── piezas pesadas, cargadas bajo demanda ───────────────────────────
    @property
    def B(self) -> sp.csr_matrix:
        if self._B is None:
            self._B = sp.load_npz(self.artifacts.operador / "B.npz").tocsr()
        return self._B

    @property
    def spectral(self) -> SpectralOps:
        """``S`` y ``Vᵀ`` en RAM (17 GB); los productos corren en CPU."""
        if self._ops is None:
            t0 = time.time()
            s = np.load(self.artifacts.svd / "S.npy")
            vt = np.load(self.artifacts.svd / "Vt.npy")
            self._ops = SpectralOps(s, vt, spectral_device="cpu")
            print(f"[drus] SVD en RAM en {time.time()-t0:.0f} s", flush=True)
        return self._ops

    @property
    def model(self) -> torch.nn.Module:
        if self._model is None:
            self._model = load_prior(self.artifacts.prior, device=self.device)
        return self._model

    @property
    def n_parameters(self) -> int:
        return n_parameters(self.model)

    # ── cadena de inferencia ────────────────────────────────────────────
    def observation(self, rf: PicmusRF, phantom: Optional[str] = None,
                    tgc_alpha: Optional[float] = None,
                    norm_scale: Optional[float] = None) -> np.ndarray:
        """``By``: TGC, normalización y conformador apodizado del artículo.

        La TGC y la escala de normalización dependen del fantoma, como en el
        código original; para datos propios se dan explícitamente.
        """
        return compute_by(self.B, rf, int(self.meta["K"]), int(self.meta["k_min"]),
                          phantom or "datos propios", tgc_alpha=tgc_alpha,
                          norm_scale=norm_scale)[0]

    def spectral_observation(self, by: np.ndarray,
                             phantom: Optional[str] = None) -> np.ndarray:
        """``Uᵀ·By``. Reutiliza el precalculado si ``By`` es el del fantoma
        canónico (salvo redondeo de coma flotante: recalcular ``By`` en otra
        máquina difiere en ~1e-13); si no, lo calcula leyendo ``U`` (17 GB) por
        mapeo de memoria.
        """
        sv, op = self.artifacts.svd, self.artifacts.operador
        if phantom is not None:
            p_uty, p_by = sv / f"Uty_{phantom}.npy", op / f"By_{phantom}.npy"
            if p_uty.exists() and p_by.exists():
                ref = np.load(p_by)
                if ref.shape == by.shape and np.allclose(
                        by, ref, rtol=1e-9, atol=1e-12 * float(np.abs(ref).max())):
                    return np.load(p_uty)
        p_u = sv / "U.npy"
        if not p_u.exists():
            raise FileNotFoundError(
                f"Para una observación distinta de las precalculadas hace falta "
                f"{p_u} (la matriz U de la SVD).")
        u = np.load(p_u, mmap_mode="r")
        return np.asarray(u.T @ by.astype(np.float32), dtype=np.float32)

    def reconstruct(self, rf: PicmusRF, phantom: Optional[str] = None,
                    gamma: Optional[float] = None, seed: int = 0,
                    tgc_alpha: Optional[float] = None,
                    norm_scale: Optional[float] = None) -> DRUSResult:
        """Reconstruye desde el RF del ángulo 0° (``PicmusRF``).

        ``phantom`` selecciona la TGC, la escala y el ``γ`` de la tesis; sin
        él hay que dar ``gamma``, ``tgc_alpha`` y ``norm_scale``.
        """
        if phantom is None and None in (gamma, tgc_alpha, norm_scale):
            raise ValueError("sin `phantom` hay que indicar gamma, tgc_alpha y norm_scale")
        if phantom is not None and phantom not in PHANTOMS:
            raise ValueError(f"phantom debe ser uno de {list(PHANTOMS)}, no {phantom!r}")
        g = THESIS_GAMMA[phantom] if gamma is None else float(gamma)
        by = self.observation(rf, phantom, tgc_alpha, norm_scale)
        uty = self.spectral_observation(by, phantom)
        res = ddrm_sample(self.model, BETAS, self.spectral, expand_uty(uty),
                          gamma=max(g, 0.1), image_size=int(self.meta["nx"]),
                          timesteps=self.timesteps, eta=self.eta,
                          eta_b=self.eta_b, seed=int(seed),
                          model_device=self.device, verbose=False)
        return DRUSResult(x=res.x_mean, bmode=env_log01(res.x_mean),
                          z_axis=self.z_axis, x_axis=self.x_axis,
                          phantom=phantom, gamma=g, seed=int(seed),
                          seconds=res.seconds)

    def reconstruct_picmus(self, hdf5_path: Union[str, Path],
                           phantom: Optional[str] = None,
                           gamma: Optional[float] = None,
                           seed: int = 0) -> DRUSResult:
        """Reconstruye directamente desde un fichero RF de PICMUS.

        El fantoma se deduce del nombre del fichero si no se indica.
        """
        phantom = phantom or phantom_from_filename(hdf5_path)
        rf = load_picmus_rf_file(Path(hdf5_path))
        if abs(rf.fs - float(self.meta["fs"])) >= 1 or abs(rf.c - float(self.meta["c"])) >= 1e-6:
            raise ValueError("la geometría del fichero no coincide con la del operador")
        return self.reconstruct(rf, phantom=phantom, gamma=gamma, seed=seed)

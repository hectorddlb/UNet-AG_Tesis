# -*- coding: utf-8 -*-
"""API pública de inferencia del método propuesto (UNet-AG).

Encapsula la cadena completa —adjunto Fd-F, mapa de F-number, red y
refinamiento residual— detrás de una interfaz estable, de modo que el modelo
pueda usarse desde otra máquina sin conocer la organización interna del
paquete ni las funciones privadas de los evaluadores.

Uso mínimo
----------
>>> from unet_ag.inference import UNetAG
>>> modelo = UNetAG.from_pretrained()             # semilla 42 por defecto
>>> bmode = modelo.reconstruct_picmus("resolution_distorsion_expe_dataset_rf.hdf5")
>>> bmode.shape
(612, 388)

Desde RF propio
---------------
>>> geom = Geometry(pitch_m=300e-6, n_elements=128, sampling_freq_hz=20.832e6,
...                 speed_of_sound=1540.0, n_samples=1527, fc=5.208e6)
>>> bmode = modelo.reconstruct(rf, geom)          # rf: (n_samples, n_elements)

La salida es siempre la envolvente comprimida logarítmicamente y normalizada
al intervalo [0, 1] sobre la rejilla de evaluación, lista para visualizarse con
un rango dinámico de 60 dB.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch import Tensor

from .models.b1_v3_residual import B1V3Residual
from .physics.f_number_map import compute_f_number_map_band_avg
from .physics.h_fdf_wrapper import make_op_for_angle
from .train.b1_attention_gate import _build_grid, _envelope_log_compress

__all__ = ["Geometry", "UNetAG", "DYN_RANGE_DB", "to_decibels"]

DYN_RANGE_DB = 60.0
"""Rango dinámico con el que se comprime la envolvente, en decibeles."""

_SEEDS = ("s42", "s1", "s2")


_CKPT_SUBPATH = Path("metodos") / "05_unet_ag_propuesto" / "checkpoints"


def _candidate_checkpoint_dirs() -> list[Path]:
    """Lugares donde buscar los pesos, en orden de prioridad.

    Los checkpoints no se empaquetan dentro de la distribución (41 MB cada uno,
    y ya viven en el repositorio), así que se localizan en tiempo de ejecución:

    1. ``$UNETAG_CHECKPOINTS`` — control explícito, gana siempre.
    2. Relativo al paquete — funciona con una instalación editable
       (``pip install -e .``) desde un clon del repositorio.
    3. Relativo al directorio de trabajo — funciona al ejecutar desde un clon
       sin instalar nada.
    """
    import os
    out: list[Path] = []
    env = os.environ.get("UNETAG_CHECKPOINTS")
    if env:
        out.append(Path(env).expanduser())
    here = Path(__file__).resolve()
    out.append(here.parents[2] / _CKPT_SUBPATH)      # <repo>/codigo/unet_ag -> <repo>
    out.append(here.parents[1] / _CKPT_SUBPATH)
    cwd = Path.cwd()
    out += [cwd / _CKPT_SUBPATH, cwd.parent / _CKPT_SUBPATH]
    return out


def _resolve_checkpoint(seed: str) -> Path:
    name = f"anchor05_{seed}.pt"
    tried = []
    for d in _candidate_checkpoint_dirs():
        p = d / name
        tried.append(str(p))
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No se encontró el checkpoint '{name}'. Se buscó en:\n  "
        + "\n  ".join(tried)
        + "\n\nSoluciones:\n"
        "  · ejecuta desde un clon del repositorio, o instálalo con "
        "`pip install -e .` desde él;\n"
        "  · o define UNETAG_CHECKPOINTS=/ruta/a/los/checkpoints;\n"
        "  · o pasa la ruta directamente: UNetAG(checkpoint='...') "
        "/ `unetag reconstruct --ckpt ...`."
    )


@dataclass(frozen=True)
class Geometry:
    """Geometría de adquisición necesaria para construir el operador físico.

    Todos los valores en unidades del SI. Los tres últimos campos tienen
    valores por omisión correspondientes al transductor de PICMUS (L11-4v).
    """

    n_elements: int
    n_samples: int
    pitch_m: float
    sampling_freq_hz: float
    speed_of_sound: float = 1540.0
    fc: float = 5.208e6
    element_width_m: float = 270e-6
    t_start: float = 0.0
    """Instante de la primera muestra de RF. Cero en PICMUS; distinto de cero
    en simulaciones de Field II, donde ``calc_scat`` devuelve un retardo."""

    def as_dict(self) -> dict:
        return asdict(self)


def to_decibels(bmode: np.ndarray, dyn_range_db: float = DYN_RANGE_DB) -> np.ndarray:
    """Convierte la salida en [0, 1] a decibeles en [-dyn_range_db, 0]."""
    return np.asarray(bmode) * dyn_range_db - dyn_range_db


class UNetAG:
    """Reconstructor B-mode desde una sola onda plana.

    Parameters
    ----------
    checkpoint : ruta al fichero de pesos (``.pt``).
    device : ``"cuda"``, ``"cpu"`` o ``"auto"`` (por omisión, GPU si la hay).
    nz, nx : dimensiones de la rejilla de evaluación. Los valores por omisión
        (612 x 388) son los de la tesis; cambiarlos altera las métricas.
    """

    def __init__(self, checkpoint: Union[str, Path],
                 device: str = "auto", nz: int = 612, nx: int = 388):
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.exists():
            raise FileNotFoundError(
                f"No se encontró el checkpoint: {self.checkpoint}\n"
                f"Descárgalo del repositorio de la tesis o indica la ruta con "
                f"UNetAG(checkpoint=...)."
            )
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") \
            if device == "auto" else device
        self.nz, self.nx = int(nz), int(nx)

        state = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        self.model = B1V3Residual(base_ch=32, embed_ch=16).to(self.device).eval()
        self.model.load_state_dict(state, strict=True)

    # ── constructores ────────────────────────────────────────────────────
    @classmethod
    def from_pretrained(cls, seed: str = "s42", **kwargs) -> "UNetAG":
        """Carga uno de los tres modelos entrenados de la tesis.

        ``seed`` ∈ {``"s42"``, ``"s1"``, ``"s2"``}. El modelo principal es
        ``s42``; los números
        publicados son la media de los tres.
        """
        if seed not in _SEEDS:
            raise ValueError(f"seed debe ser uno de {_SEEDS}, no {seed!r}")
        return cls(_resolve_checkpoint(seed), **kwargs)

    # ── propiedades ──────────────────────────────────────────────────────
    @property
    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    # ── inferencia ───────────────────────────────────────────────────────
    def grid(self, geometry: Geometry) -> tuple[np.ndarray, np.ndarray]:
        """Rejilla de evaluación en metros: ``(z, x)``."""
        gz, gx = _build_grid(geometry.as_dict(), self.nz, self.nx)
        return gz.numpy(), gx.numpy()

    @torch.no_grad()
    def adjoint(self, rf: Union[np.ndarray, Tensor],
                geometry: Geometry) -> np.ndarray:
        """Solución inicial: el adjunto Fd-F, en envolvente log [0, 1].

        Es a la vez la entrada de la red y la imagen de anclaje del método, y
        por sí sola constituye una reconstrucción clásica utilizable.
        """
        rf_t = self._prepare_rf(rf, geometry)
        gz, gx = _build_grid(geometry.as_dict(), self.nz, self.nx)
        op = make_op_for_angle(0.0, geometry.as_dict(), gz, gx,
                               device=self.device, nf_chunk=None)
        x0 = op.adjoint(rf_t.to(self.device).to(op.float_dtype))
        if x0.is_complex():
            x0 = x0.real
        return _envelope_log_compress(x0.float().cpu()).numpy()

    @torch.no_grad()
    def reconstruct(self, rf: Union[np.ndarray, Tensor], geometry: Geometry,
                    return_adjoint: bool = False):
        """Reconstruye el B-mode desde el RF de una sola onda plana.

        Parameters
        ----------
        rf : ``(n_samples, n_elements)``, RF crudo del ángulo no direccionado.
        geometry : geometría de la adquisición.
        return_adjoint : si es ``True`` devuelve ``(bmode, adjunto)``.

        Returns
        -------
        ``(nz, nx)`` float32 en [0, 1]: envolvente comprimida a 60 dB.
        """
        x0_env = self.adjoint(rf, geometry)
        gz, gx = _build_grid(geometry.as_dict(), self.nz, self.nx)
        f_map = compute_f_number_map_band_avg(gz.numpy(), gx.numpy(),
                                              geometry.as_dict())
        x0_t = torch.from_numpy(x0_env).to(self.device)[None, None]
        f_t = torch.from_numpy(f_map).to(self.device)[None, None]
        pred = self.model(x0_t, f_t)[0, 0].cpu().numpy().astype(np.float32)
        return (pred, x0_env) if return_adjoint else pred

    def reconstruct_picmus(self, hdf5_path: Union[str, Path],
                           angle_index: Optional[int] = None,
                           return_adjoint: bool = False):
        """Reconstruye directamente desde un fichero RF de PICMUS.

        Selecciona el ángulo no direccionado (0°) salvo que se indique otro
        con ``angle_index``.
        """
        rf, geom = load_picmus_rf(hdf5_path, angle_index=angle_index)
        return self.reconstruct(rf, geom, return_adjoint=return_adjoint)

    # ── interno ──────────────────────────────────────────────────────────
    @staticmethod
    def _prepare_rf(rf: Union[np.ndarray, Tensor], geometry: Geometry) -> Tensor:
        t = torch.as_tensor(np.asarray(rf)) if not isinstance(rf, Tensor) else rf
        t = t.float()
        if t.ndim != 2:
            raise ValueError(f"rf debe ser 2-D (n_samples, n_elements); "
                             f"recibido {tuple(t.shape)}")
        if t.shape[1] != geometry.n_elements:
            raise ValueError(
                f"rf tiene {t.shape[1]} canales pero la geometría declara "
                f"n_elements={geometry.n_elements}. ¿Está transpuesto?")
        # Normalización z-score, como en entrenamiento y evaluación.
        return (t - t.mean()) / (t.std() + 1e-8)


def load_picmus_rf(hdf5_path: Union[str, Path],
                   angle_index: Optional[int] = None
                   ) -> tuple[np.ndarray, Geometry]:
    """Lee un fichero RF de PICMUS y devuelve ``(rf_1pw, Geometry)``.

    Requiere ``h5py``. El ángulo por omisión es el más próximo a 0°.
    """
    try:
        import h5py
    except ImportError as exc:                       # pragma: no cover
        raise ImportError("Leer PICMUS requiere h5py: pip install h5py") from exc

    with h5py.File(hdf5_path, "r") as f:
        g = f["US"]["US_DATASET0000"]
        data = np.asarray(g["data"]["real"])         # (n_angles, n_el, n_samples)
        angles = np.asarray(g["angles"]).reshape(-1)
        fs = float(np.asarray(g["sampling_frequency"]).reshape(-1)[0])
        c = float(np.asarray(g["sound_speed"]).reshape(-1)[0])
        fc = float(np.asarray(g["modulation_frequency"]).reshape(-1)[0]) or 5.208e6
        probe = np.asarray(g["probe_geometry"])      # (3, n_el) o (n_el, 3)
        t0 = float(np.asarray(g["initial_time"]).reshape(-1)[0])

    if probe.shape[0] == 3:
        xs = probe[0, :]
    else:
        xs = probe[:, 0]
    pitch = float(np.abs(np.diff(xs)).mean())
    idx = int(np.argmin(np.abs(angles))) if angle_index is None else int(angle_index)
    rf = np.asarray(data[idx]).T                     # -> (n_samples, n_elements)

    geom = Geometry(
        n_elements=rf.shape[1], n_samples=rf.shape[0], pitch_m=pitch,
        sampling_freq_hz=fs, speed_of_sound=c, fc=fc, t_start=t0,
    )
    return rf, geom

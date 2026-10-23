"""Dataset de fine-tuning del prior DDRM — fairness estricta con anchor05.

Las imágenes son EXACTAMENTE la supervisión que vio B1/anchor05: los targets
Schiffner_75PW `compounded.npy` (287 phantoms points; cysts no tiene targets),
procesados con la misma cadena de `b1_v4_anchored.py`:

    raw (612×388 nativo o resize bilinear a él) → hilbert axial → envelope
    → log-compress 60 dB → [0,1]

y después llevados al dominio del prior de difusión:

    resize bilinear a 256×256 → escala [−1,1] → replicado a 3 canales (RGB).

Augment en train: solo flip horizontal (consistente con la simetría L/R de la
física del transductor).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import hilbert
from torch.utils.data import Dataset

DYN_RANGE_DB = 60.0
NATIVE_NZ, NATIVE_NX = 612, 388


def envelope_log_compress(arr: np.ndarray, dyn_range_db: float = DYN_RANGE_DB) -> np.ndarray:
    """Réplica exacta de unet_ag.train.b1_attention_gate._envelope_log_compress."""
    env = np.abs(hilbert(arr, axis=0))
    env_max = float(env.max()) + 1e-12
    db = 20.0 * np.log10(env / env_max + 1e-30)
    db = np.clip(db, -dyn_range_db, 0.0)
    return ((db + dyn_range_db) / dyn_range_db).astype(np.float32)   # [0, 1]


def _resize(arr: np.ndarray, nz: int, nx: int) -> np.ndarray:
    t = torch.from_numpy(arr.astype(np.float32))[None, None]
    t = F.interpolate(t, size=(nz, nx), mode="bilinear", align_corners=False)
    return t[0, 0].numpy()


def process_target(raw: np.ndarray, size: int = 256) -> np.ndarray:
    """compounded.npy crudo → imagen env_log [0,1] de size×size (cadena B1 + resize)."""
    if raw.shape != (NATIVE_NZ, NATIVE_NX):
        raw = _resize(raw, NATIVE_NZ, NATIVE_NX)
    img01 = envelope_log_compress(raw)
    return _resize(img01, size, size)


def process_target_signed(raw: np.ndarray, size: int = 256) -> np.ndarray:
    """compounded.npy crudo → reflectividad CON SIGNO en [−1,1] de size×size.

    Dominio correcto del prior DDRM: las salidas x de Zhang (verificadas en los
    .mat del repo, BH50_fineTune) son cero-media/simétricas — el prior modela
    reflectividad firmada, consistente con y = Hx. Normalización por máximo
    absoluto por imagen. Caveat documentado: el resize bilinear 612→256 muestrea
    el carrier axial a ~2.1 px/λ (borde de Nyquist).
    """
    signed = _resize(raw.astype(np.float32), size, size)
    return (signed / (np.abs(signed).max() + 1e-30)).astype(np.float32)


def build_ft_dataset(target_root: Path, out_dir: Path, size: int = 256,
                     domain: str = "signed") -> int:
    proc = {"signed": process_target_signed, "envlog": process_target}[domain]
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for sub in sorted(target_root.glob("*/*/compounded.npy")):
        phantom_id = sub.parent.name                     # e.g. points_0001
        raw = np.load(sub)
        np.save(out_dir / f"{phantom_id}.npy", proc(raw, size))
        n += 1
    return n


class SchiffnerFTDataset(Dataset):
    """Yields (img (3,S,S) float32 en [−1,1], {}) — protocolo guided-diffusion.

    ``domain='signed'``: los .npy ya están en [−1,1] (reflectividad firmada).
    ``domain='envlog'``: los .npy están en [0,1] → se mapean a [−1,1].
    En signed, el hflip se acompaña de un flip de signo aleatorio global
    (simetría física del speckle: x y −x explican el mismo B-mode).
    """

    def __init__(self, data_dir: Path, hflip: bool = True, domain: str = "signed"):
        self.files = sorted(Path(data_dir).glob("*.npy"))
        if not self.files:
            raise FileNotFoundError(f"sin .npy en {data_dir}")
        self.hflip = hflip
        self.domain = domain

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        img = np.load(self.files[idx])
        if self.hflip and np.random.rand() < 0.5:
            img = img[:, ::-1].copy()
        if self.domain == "envlog":
            img = img * 2.0 - 1.0
        elif self.domain == "signed" and np.random.rand() < 0.5:
            img = -img
        img3 = np.repeat(img[None, :, :], 3, axis=0).astype(np.float32)
        return torch.from_numpy(img3), {}

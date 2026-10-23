"""Prior de difusión de DRUS: configuración y carga del modelo.

El método comparado en la tesis usa el prior **sin ajuste fino**: el modelo
incondicional 256×256 de guided-diffusion entrenado en ImageNet que publicó
OpenAI (``256x256_diffusion_uncond.pt``, 552.8 M parámetros). Es la variante de
mejor desempeño de la réplica (Capítulo 6 de la tesis).

``MODEL_FLAGS`` y ``DIFFUSION_FLAGS`` son los del checkpoint publicado
(= ``configs/imagenet_256.yml`` del repositorio DRUS-v1); si se cambian, el
``state_dict`` deja de cargar. Este módulo es la única fuente de esos valores:
los scripts de ajuste fino, de muestreo y de conteo de operaciones los importan
de aquí.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import torch

__all__ = ["MODEL_FLAGS", "DIFFUSION_FLAGS", "PRIOR_FILENAME", "PRIOR_URL",
           "PRIOR_SIZE_BYTES", "PRIOR_MD5", "load_prior", "n_parameters"]

PRIOR_FILENAME = "256x256_diffusion_uncond.pt"
PRIOR_URL = ("https://openaipublic.blob.core.windows.net/diffusion/jul-2021/"
             + PRIOR_FILENAME)
PRIOR_SIZE_BYTES = 2_211_383_297
PRIOR_MD5 = "fd9dd2335b8736d521de0aed54bd90ca"

MODEL_FLAGS = dict(
    image_size=256,
    num_channels=256,
    num_res_blocks=2,
    channel_mult="",
    learn_sigma=True,
    class_cond=False,
    use_checkpoint=True,          # gradient checkpointing: necesario en 24 GB
    attention_resolutions="32,16,8",
    num_heads=4,
    num_head_channels=64,
    num_heads_upsample=-1,
    use_scale_shift_norm=True,
    dropout=0.0,
    resblock_updown=True,
    use_fp16=True,
    use_new_attention_order=False,
)
DIFFUSION_FLAGS = dict(
    diffusion_steps=1000,
    noise_schedule="linear",
    timestep_respacing="",
    use_kl=False,
    predict_xstart=False,
    rescale_timesteps=False,
    rescale_learned_sigmas=False,
)


def load_prior(checkpoint: Union[str, Path], device: str = "cuda",
               fp16: Union[bool, None] = None) -> torch.nn.Module:
    """Instancia la UNet de guided-diffusion y carga los pesos del prior.

    Reproduce la carga con la que se generaron los resultados de la tesis
    (``metodos/04_drus/scripts/sample_ddrm.py``): pesos en ``float32`` y
    conversión a ``float16`` de los bloques convolucionales, en modo
    evaluación. ``fp16=None`` aplica esa conversión solo en GPU; en CPU el
    modelo se queda en ``float32`` (y el resultado deja de ser idéntico al de
    la tesis).
    """
    from guided_diffusion.script_util import create_model_and_diffusion

    flags = dict(MODEL_FLAGS)
    if fp16 is None:
        fp16 = torch.device(device).type == "cuda"
    flags["use_fp16"] = bool(fp16)
    model, _ = create_model_and_diffusion(**flags, **DIFFUSION_FLAGS)
    state = torch.load(Path(checkpoint), map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    if flags["use_fp16"]:
        model.convert_to_fp16()
    model.eval()
    return model


def n_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

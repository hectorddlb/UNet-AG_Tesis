"""
Reemplazo single-process (sin MPI) del dist_util original de openai/guided-diffusion.

El trainer usa torch.distributed para all-reduce/broadcast; aquí inicializamos
un process group de 1 proceso para que esas llamadas sean no-ops válidas,
sin depender de mpi4py ni blobfile.
"""

import os
import socket

import torch as th
import torch.distributed as dist

GPUS_PER_NODE = 1


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def setup_dist():
    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(_find_free_port())
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    backend = "nccl" if th.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, rank=0, world_size=1)


def dev():
    if th.cuda.is_available():
        return th.device("cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    return th.load(path, **kwargs)


def sync_params(params):
    # 1 proceso: nada que sincronizar
    return

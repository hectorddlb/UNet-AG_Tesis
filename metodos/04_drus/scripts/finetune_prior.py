"""Fase C.3 — fine-tune del prior de difusión (guided-diffusion 256×256
ImageNet uncond) con las imágenes Schiffner75 de anchor05.

Réplica del procedimiento de Zhang et al. 2023 (checkpoint model000000.pt →
model006000.pt, es decir 6000 steps de FT con imágenes US no pareadas), pero
con NUESTRO dataset de fairness (287 Schiffner75 env_log en vez de sus 800
CIRS/TPAC).

Uso (sbatch):
    python zhang_drus/scripts/finetune_prior.py \
        --data_dir zhang_drus/results_drus/ft_data_schiffner287 \
        --base_ckpt zhang_drus/results_drus/ckpt/256x256_diffusion_uncond.pt \
        --out_dir zhang_drus/results_drus/ft_prior_schiffner287 \
        --max_steps 6000 --batch_size 4 --microbatch 1 --lr 1e-5

Reanudación tras límite de 18 h: re-lanzar igual; detecta el último
model??????.pt en out_dir y continúa (optimizer + EMA incluidos).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3] / "codigo"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "zhang_drus"))   # guided_diffusion importable plano

from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import create_model_and_diffusion
from guided_diffusion.train_util import TrainLoop
from drus.ft_dataset import SchiffnerFTDataset

# Flags EXACTOS del checkpoint 256x256_diffusion_uncond.pt (= configs/imagenet_256.yml
# del repo DRUS-v1): viven en drus.prior, la única fuente del proyecto.
from drus.prior import DIFFUSION_FLAGS, MODEL_FLAGS  # noqa: E402


class FTTrainLoop(TrainLoop):
    """TrainLoop de openai con tope duro de steps (sin anneal de lr)."""

    def __init__(self, *args, max_steps: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_steps = max_steps

    def run_loop(self):
        while self.step + self.resume_step < self.max_steps:
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0 and self.step > 0:
                self.save()
            self.step += 1                      # como el run_loop original de openai
        self.save()


def _find_resume(out_dir: Path) -> Path | None:
    ckpts = sorted(out_dir.glob("model??????.pt"))
    return ckpts[-1] if ckpts else None


def _infinite(loader: DataLoader):
    while True:
        yield from loader


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, required=True)
    ap.add_argument("--base_ckpt", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--max_steps", type=int, default=6000)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--microbatch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--ema_rate", type=str, default="0.9999")
    ap.add_argument("--log_interval", type=int, default=20)
    ap.add_argument("--save_interval", type=int, default=1000)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--domain", choices=["signed", "envlog"], default="signed",
                    help="signed = reflectividad [−1,1] (dominio físico de x, default)")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.environ["OPENAI_LOGDIR"] = str(args.out_dir)
    dist_util.setup_dist()
    logger.configure(dir=str(args.out_dir))

    # Resume: último model??????.pt del out_dir; si no hay, sembrar el base
    # ckpt como model000000.pt (convención de Zhang) para arrancar el FT.
    resume = _find_resume(args.out_dir)
    if resume is None:
        seed_ckpt = args.out_dir / "model000000.pt"
        if not seed_ckpt.exists():
            logger.log(f"seed: {args.base_ckpt} → {seed_ckpt}")
            shutil.copy2(args.base_ckpt, seed_ckpt)
        resume = seed_ckpt
    logger.log(f"resume_checkpoint = {resume}")

    model, diffusion = create_model_and_diffusion(**MODEL_FLAGS, **DIFFUSION_FLAGS)
    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler("uniform", diffusion)

    ds = SchiffnerFTDataset(args.data_dir, hflip=True, domain=args.domain)
    logger.log(f"dataset: {len(ds)} imágenes ({args.domain}) de {args.data_dir}")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True, pin_memory=True)

    FTTrainLoop(
        model=model,
        diffusion=diffusion,
        data=_infinite(loader),
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=str(resume),
        use_fp16=MODEL_FLAGS["use_fp16"],
        fp16_scale_growth=1e-3,
        schedule_sampler=schedule_sampler,
        weight_decay=0.0,
        lr_anneal_steps=0,
        max_steps=args.max_steps,
    ).run_loop()
    logger.log("[done] fine-tune completo")


if __name__ == "__main__":
    main()

"""Coste computacional por imagen de los 6 metodos de la tesis.

FLOPs analiticos (independientes del hardware). El tiempo de pared se mide
aparte, con tiempo_inferencia.py.

POR QUE NO BASTA UN CONTADOR AUTOMATICO
---------------------------------------
``torch.utils.flop_counter.FlopCounterMode`` solo ve conv/matmul/bmm/sdpa. Los
tres metodos clasicos (das1pw, das75pw, schiffner75pw) y el adjunto que alimenta
a UNet-AG son **elementwise**: un contador automatico los reporta como 0 FLOPs.
Hay que contarlos a mano sobre el kernel real. Las dos redes si van con el
contador automatico.

QUE SE CUENTA EXACTAMENTE
-------------------------
Se cuenta lo que el protocolo de evaluacion ejecuta de verdad, que NO es lo
que sugiere el nombre de cada metodo:

  das1pw, schiffner1pw, das75pw, schiffner75pw -> los cuatro usan
      ``schiffner_das.das_pw``, el beamformer de Fourier (ver
      gen_classical_picmus_fixed.py). Solo cambian ``ConstantFNumber(0)`` por
      ``GratingAngleLB(60,1.5)`` y el numero de angulos. Es decir: DAS y
      Schiffner Fd-F cuestan lo mismo, y la apodizacion Fd-F es gratis frente
      al compounding.
  unet_ag  -> adjunto ``OpHFdf.adjoint`` (1 angulo, mismo kernel de Fourier,
      banda mas ancha) + forward del UNet+AG.
  drus     -> 50 pasos DDRM, cada uno con 1 forward del UNet 256x256 de
      guided-diffusion + 3 productos por V/Vt (65536 x 65536).

Se anade como referencia el DAS **temporal** (``physics/classical/das_1pw.py``),
que es lo que hace un ecografo real, para que la comparacion con la literatura
tenga sentido.

CONVENCION DE FLOPs
-------------------
  suma/resta/producto/division real .... 1 FLOP
  sqrt, sin, cos, exp .................. 1 FLOP  (se reportan aparte para
                                         que el lector pueda repesarlos)
  producto complejo .................... 6 FLOPs (4 mul + 2 add)
  suma compleja ........................ 2 FLOPs
  exp(i*phi) ........................... 2 FLOPs (sin + cos)
  matmul (m,k)x(k,n) ................... 2*m*k*n

POR ESCENARIO
-------------
El numero de frecuencias que procesan los conformadores de Fourier depende del
numero de muestras de cada adquisicion, asi que el coste se calcula para los
cuatro fantomas de PICMUS (``coste_por_escenario.csv``). Hay dos
bandas: la de los metodos clasicos (2.25-6.75 MHz, la de ``das_pw``) y la del
adjunto que alimenta a UNet-AG (0.5-1.5 fc = 2.604-7.812 MHz, la de ``OpHFdf``),
mas ancha, por lo que el adjunto cuesta mas que el DAS de 1 PW. La hipotesis se
enuncia sobre los escenarios experimentales: la lista ``metodos`` del JSON, que
alimenta la comparacion de coste, es la de ese escenario.

Uso
---
    python evaluacion/scripts/coste_computacional.py      # PICMUS en $TESIS_DATASETS/PICMUS
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
DATASETS = Path(os.environ.get("TESIS_DATASETS", ROOT / "datos"))  # PICMUS/ y Field II/, ver datos/README.md
sys.path.insert(0, str(ROOT / "codigo"))

# Rejilla canonica de evaluacion
NZ, NX = 612, 388
N_ANGLES = 75
DRUS_TIMESTEPS = 50
DRUS_SIDE = 256
DRUS_CHANNELS = 3


# ---------------------------------------------------------------------------
# 1. Dimensiones reales del problema (leidas de PICMUS, no inventadas)
# ---------------------------------------------------------------------------

SCENARIOS = {
    "sim_resolution": "simulation/resolution_distorsion/resolution_distorsion_simu_dataset_rf.hdf5",
    "sim_contrast":   "simulation/contrast_speckle/contrast_speckle_simu_dataset_rf.hdf5",
    "exp_resolution": "experiments/resolution_distorsion/resolution_distorsion_expe_dataset_rf.hdf5",
    "exp_contrast":   "experiments/contrast_speckle/contrast_speckle_expe_dataset_rf.hdf5",
}
SCENARIO_HYPOTHESIS = "exp_resolution"   # = exp_contrast: mismas dimensiones


def picmus_dims(rf_rel: str, band: str = "das") -> dict:
    """Nt, Nel, fs, banda y Nf tal y como los derivan ``das_pw`` y ``OpHFdf``.

    ``band="das"``: banda del dataset (2.25-6.75 MHz), la de los conformadores
    clasicos. ``band="adjunto"``: 0.5-1.5 fc con fc = fs/4, el valor por
    omision de ``OpHFdf`` (measurement_operator_fdf.py), la del adjunto.
    """
    from schiffner_das import load_picmus_hdf5

    ds = load_picmus_hdf5(DATASETS / "PICMUS" / "database" / rf_rel)
    rf = ds.rf_for_angle(0)
    Nt, Nel = rf.shape
    fs, c = float(ds.f_s), float(ds.c_avg)
    pitch = float(ds.element_pitch)
    if band == "das":
        f_lb, f_ub = float(ds.f_lb), float(ds.f_ub)
    else:
        fc = fs / 4.0
        f_lb, f_ub = 0.5 * fc, 1.5 * fc

    # Zero-padding y eje de frecuencia: replica exacta de das_pw (lineas 178-196)
    M = (Nel - 1) / 2.0
    ctr_first, ctr_last = -M * pitch, M * pitch
    span = ctr_last - ctr_first
    # z0 de la rejilla canonica de evaluacion
    z0 = 5e-3
    Npad = math.ceil((math.sqrt(span ** 2 + z0 ** 2) - z0) * fs / c)
    Ndft = Nt + Npad + 1 - (Nt + Npad) % 2
    ifl = math.ceil(f_lb * Ndft / fs)
    ifu = math.floor(f_ub * Ndft / fs)
    Nf = ifu - ifl + 1

    return dict(Nt=Nt, Nel=Nel, fs=fs, c=c, pitch=pitch,
                f_lb=f_lb, f_ub=f_ub,
                Npad=Npad, Ndft=Ndft, Nf=Nf,
                n_angles=int(len(ds.steering_angles_rad)))


# ---------------------------------------------------------------------------
# 2. FLOPs analiticos
# ---------------------------------------------------------------------------

# Coste por tupla (frecuencia, pixel, elemento) del kernel de Fourier.
# Derivado linea a linea de beamforming.py:264-315 / measurement_operator_fdf.py:439-451
FOURIER_KERNEL_COST = {
    "aw * tof":                    1,   # mul real
    "exp(1j * .)":                 2,   # sin + cos
    "dlat_a / hw_elem":            1,   # div (p_hw)
    "Tukey.compute_samples":       5,   # 2 cmp + sub + div + mul + cos + add + div
    "apod * valid":                1,
    "NormalizationOn (sum + div)": 2,
    "apod(real) * eph(complejo)":  2,
    "* rf_bp (complejo x complejo)": 6,
    "sum sobre elementos":         2,   # suma compleja
    "sum sobre frecuencias":       2,   # suma compleja
}
FOURIER_PER_TUPLE = sum(FOURIER_KERNEL_COST.values())
FOURIER_TRANSC_PER_TUPLE = 3          # sin, cos (exp) + cos (Tukey)

# Coste por tupla (pixel, elemento) del DAS temporal (das_1pw.py:44-77)
TIME_DAS_COST = {
    "dx = x - xk":            1,
    "dx*dx + z*z":            2,
    "sqrt":                   1,
    "/ c":                    1,
    "+ t_tx - t_start":       2,
    "* fs":                   1,
    "round":                  1,
    "in_range (2 cmp)":       2,
    "in_aper (abs + cmp)":    2,
    "Hanning (mul+div+cos+add+mul)": 5,
    "clip":                   2,
    "samples * v * w":        2,
    "acumulacion":            1,
}
TIME_DAS_PER_TUPLE = sum(TIME_DAS_COST.values())
TIME_DAS_TRANSC_PER_TUPLE = 2         # sqrt + cos


def fourier_kernel_flops(d: dict, nz: int, nx: int, n_angles: int) -> dict:
    """FLOPs de ``n_angles`` pasadas del beamformer de Fourier."""
    tuples = d["Nf"] * nz * nx * d["Nel"] * n_angles
    focus = tuples * FOURIER_PER_TUPLE
    # FFT por elemento y angulo: 5 N log2 N (convencion habitual para FFT compleja)
    fft = n_angles * d["Nel"] * 5.0 * d["Ndft"] * math.log2(d["Ndft"])
    # Suma coherente entre angulos (complejo)
    compound = 2.0 * nz * nx * (n_angles - 1) if n_angles > 1 else 0.0
    return {
        "total": focus + fft + compound,
        "focus": focus,
        "fft": fft,
        "compound": compound,
        "tuples": tuples,
        "transcendentales": tuples * FOURIER_TRANSC_PER_TUPLE,
    }


def time_das_flops(d: dict, nz: int, nx: int, n_angles: int) -> dict:
    tuples = nz * nx * d["Nel"] * n_angles
    return {
        "total": tuples * TIME_DAS_PER_TUPLE,
        "tuples": tuples,
        "transcendentales": tuples * TIME_DAS_TRANSC_PER_TUPLE,
    }


def drus_spectral_flops() -> dict:
    """Productos por V/Vt del muestreador DDRM (ddrm_sampler.py:53-64, 120-197).

    Cada llamada es matmul (N,N) x (N,3) = 2*N*N*3 FLOPs, con N = 256^2.
    Por paso: Vt(x0_t), Vt(et), V(vt_next) = 3 llamadas. Mas 1 V(init_y) inicial.
    """
    n = DRUS_SIDE * DRUS_SIDE
    per_call = 2.0 * n * n * DRUS_CHANNELS
    n_calls = 3 * DRUS_TIMESTEPS + 1
    return {"total": per_call * n_calls, "per_call": per_call, "n_calls": n_calls}


# ---------------------------------------------------------------------------
# 3. FLOPs de las redes (contador automatico de PyTorch)
# ---------------------------------------------------------------------------

def unet_ag_flops(device: str = "cpu") -> dict:
    from torch.utils.flop_counter import FlopCounterMode
    from unet_ag.models.b1_v3_residual import B1V3Residual

    model = B1V3Residual(base_ch=32, embed_ch=16).to(device).eval()
    x0 = torch.zeros(1, 1, NZ, NX, device=device)
    f_map = torch.zeros(1, 1, NZ, NX, device=device)

    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        model(x0, f_map)
    n_par = sum(p.numel() for p in model.parameters())
    return {"total": float(counter.get_total_flops()), "params": n_par}


def drus_unet_flops(device: str = "cpu") -> dict:
    from torch.utils.flop_counter import FlopCounterMode
    from guided_diffusion.script_util import create_model_and_diffusion
    from drus.prior import DIFFUSION_FLAGS, MODEL_FLAGS

    flags = dict(MODEL_FLAGS)
    flags["use_fp16"] = False        # los FLOPs no cambian; evita convert_to_fp16
    flags["use_checkpoint"] = False  # el checkpointing re-ejecuta el forward
    model, _ = create_model_and_diffusion(**flags, **DIFFUSION_FLAGS)
    model.to(device).eval()

    x = torch.zeros(1, DRUS_CHANNELS, DRUS_SIDE, DRUS_SIDE, device=device)
    t = torch.zeros(1, device=device)

    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        model(x, t)
    n_par = sum(p.numel() for p in model.parameters())
    del model
    return {"total": float(counter.get_total_flops()), "params": n_par}


# ---------------------------------------------------------------------------
# 4. Ensamblado
# ---------------------------------------------------------------------------

def build_table(dp: dict, da: dict, ag: dict, du: dict, sp: dict) -> list[dict]:
    """Filas de la tabla de coste para un escenario.

    ``dp``: dimensiones en la banda de los conformadores clasicos; ``da``: en
    la banda del adjunto de UNet-AG.
    """
    k1 = fourier_kernel_flops(dp, NZ, NX, 1)
    k75 = fourier_kernel_flops(dp, NZ, NX, N_ANGLES)
    adj = fourier_kernel_flops(da, NZ, NX, 1)
    return [
        dict(metodo="DAS 1 PW", entrada="1 PW",
             flops=k1["total"], params=0,
             desglose=f"Fourier x1 angulo ({k1['focus']/k1['total']*100:.1f}% focus)"),
        dict(metodo="Schiffner Fd-F 1 PW", entrada="1 PW",
             flops=k1["total"], params=0,
             desglose="identico a DAS 1 PW: solo cambia el F-number, "
                      "que no anade tuplas al kernel"),
        dict(metodo="DAS 75 PW", entrada="75 PW",
             flops=k75["total"], params=0,
             desglose="Fourier x75 angulos"),
        dict(metodo="Schiffner Fd-F 75 PW", entrada="75 PW",
             flops=k75["total"], params=0,
             desglose="identico a DAS 75 PW: la Fd-F no anade tuplas"),
        dict(metodo="DRUS (DDRM 50 pasos)", entrada="1 PW",
             flops=DRUS_TIMESTEPS * du["total"] + sp["total"], params=du["params"],
             desglose=f"50 x UNet256 ({DRUS_TIMESTEPS*du['total']/1e12:.2f} TFLOP) "
                      f"+ V/Vt ({sp['total']/1e12:.2f} TFLOP)"),
        dict(metodo="UNet-AG (propuesto)", entrada="1 PW",
             flops=adj["total"] + ag["total"], params=ag["params"],
             desglose=f"adjunto ({adj['total']/1e9:.1f} GFLOP, {da['Nf']} frecuencias) "
                      f"+ UNet ({ag['total']/1e9:.1f} GFLOP)"),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--net_device", default="cpu",
                    help="donde instanciar las redes para contar FLOPs (cpu basta)")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "evaluacion/metricas/coste_computacional.json")
    ap.add_argument("--csv", type=Path,
                    default=ROOT / "evaluacion/metricas/coste_por_escenario.csv")
    args = ap.parse_args()

    print("  contando UNet-AG ...", flush=True)
    ag = unet_ag_flops(args.net_device)
    print("  contando UNet de guided-diffusion (552 M par, tarda) ...", flush=True)
    du = drus_unet_flops(args.net_device)
    sp = drus_spectral_flops()

    escenarios, csv_rows = {}, []
    for key, rf_rel in SCENARIOS.items():
        dp, da = picmus_dims(rf_rel, "das"), picmus_dims(rf_rel, "adjunto")
        rows = build_table(dp, da, ag, du, sp)
        g = {r["metodo"]: r["flops"] / 1e9 for r in rows}
        adj = fourier_kernel_flops(da, NZ, NX, 1)["total"] / 1e9
        escenarios[key] = dict(dimensiones_das=dp, dimensiones_adjunto=da, metodos=rows)
        csv_rows.append(dict(
            escenario=key, Nt=dp["Nt"], Nf_das=dp["Nf"], Nf_adjunto=da["Nf"],
            das1pw_gflop=g["DAS 1 PW"], adjunto_gflop=adj, red_gflop=ag["total"] / 1e9,
            propuesto_gflop=g["UNet-AG (propuesto)"], pw75_gflop=g["DAS 75 PW"],
            drus_gflop=g["DRUS (DDRM 50 pasos)"],
            razon_75pw=g["DAS 75 PW"] / g["UNet-AG (propuesto)"],
            razon_drus=g["DRUS (DDRM 50 pasos)"] / g["UNet-AG (propuesto)"]))
        print(f"\n=== {key}: Nt={dp['Nt']} Nf(das)={dp['Nf']} Nf(adjunto)={da['Nf']} ===")
        for r in rows:
            print(f"  {r['metodo']:24s} {r['flops']/1e9:12.3f} GFLOP  {r['desglose']}")

    hyp = escenarios[SCENARIO_HYPOTHESIS]
    td1 = time_das_flops(hyp["dimensiones_das"], NZ, NX, 1)
    td75 = time_das_flops(hyp["dimensiones_das"], NZ, NX, N_ANGLES)
    print(f"\n  [referencia] DAS temporal 1 PW  = {td1['total']/1e9:.3f} GFLOP")
    print(f"  [referencia] DAS temporal 75 PW = {td75['total']/1e9:.3f} GFLOP")

    payload = dict(
        escenario_metodos=SCENARIO_HYPOTHESIS,
        dimensiones=hyp["dimensiones_das"],
        dimensiones_adjunto=hyp["dimensiones_adjunto"],
        rejilla=[NZ, NX],
        convencion=dict(kernel_fourier_por_tupla=FOURIER_KERNEL_COST,
                        das_temporal_por_tupla=TIME_DAS_COST),
        metodos=hyp["metodos"],
        escenarios=escenarios,
        red=dict(unet_ag=ag, drus_unet=du, drus_espectral=sp),
        referencia_das_temporal=dict(pw1=td1, pw75=td75),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nwrote {args.out}")

    import csv
    with open(args.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(csv_rows[0]))
        w.writeheader()
        for r in csv_rows:
            w.writerow({k: (f"{v:.3f}" if isinstance(v, float) else v) for k, v in r.items()})
    print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()

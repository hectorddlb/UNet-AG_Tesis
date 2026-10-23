"""Tiempo de inferencia por imagen de los 6 metodos, medido en GPU.

Complementa ``coste_computacional.py`` (FLOPs analiticos). Aqui se mide reloj
de pared con ``torch.cuda.synchronize()`` alrededor de cada repeticion y un
warmup previo para no cronometrar la compilacion de kernels ni la carga
diferida de contexto CUDA.

EXTRAPOLACION DE LOS METODOS DE 75 ANGULOS
------------------------------------------
``das75pw`` y ``schiffner75pw`` repiten el MISMO kernel 75 veces con la unica
diferencia del angulo de direccion; la suma coherente entre angulos es
despreciable (2*Nz*Nx*74 FLOPs frente a 23 TFLOP). Se miden ``--n_angles``
angulos reales (por defecto 8, repartidos por el barrido para cubrir angulos
extremos y centrales) y se extrapola x75, reportando la desviacion tipica
entre angulos para que se vea que la extrapolacion es legitima. Medir los 75
costaria ~35 min de GPU por metodo.

DRUS
----
Se mide en su configuracion nativa con el paquete ``drus``: una reconstruccion
de extremo a extremo (50 pasos DDRM, prior en float16 en GPU, productos por
``Vt`` de 65536x65536 en CPU) y, aparte, los dos componentes de un paso. La
carga de ``Vt.npy`` (17.2 GB) se reporta aparte: es coste de puesta en marcha,
no de inferencia.

UNET-AG
-------
Se mide tal como se ejecuta (``unet_ag.inference``): el adjunto Fd-F con la
banda que procesa el modelo (0.5-1.5 f_c), la envolvente log (CPU) y la red.

ESCENARIO Y NODO
----------------
Por omision, el escenario experimental (``--phantom exp_resolution``), el mismo
de los GFLOP de la tabla de costo, y todos los metodos en el mismo proceso y el
mismo nodo, con ``das_pw`` en float32 (precision igualada con el adjunto).

Uso
---
    python evaluacion/scripts/tiempo_inferencia.py --device cuda
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
DATASETS = Path(os.environ.get("TESIS_DATASETS", ROOT / "datos"))  # PICMUS/ y Field II/, ver datos/README.md
DRUS_ARTIFACTS = Path(os.environ.get("DRUS_ARTIFACTS", ROOT / "metodos/04_drus/artefactos"))
sys.path.insert(0, str(ROOT / "codigo"))

PHANTOM_RF = {
    "sim_resolution": "simulation/resolution_distorsion/resolution_distorsion_simu_dataset_rf.hdf5",
    "sim_contrast":   "simulation/contrast_speckle/contrast_speckle_simu_dataset_rf.hdf5",
    "exp_resolution": "experiments/resolution_distorsion/resolution_distorsion_expe_dataset_rf.hdf5",
    "exp_contrast":   "experiments/contrast_speckle/contrast_speckle_expe_dataset_rf.hdf5",
}

NZ, NX = 612, 388
N_ANGLES_TOTAL = 75
DRUS_TIMESTEPS_TOTAL = 50
DRUS_SIDE, DRUS_CHANNELS = 256, 3


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def timeit(fn, device: str, n_warmup: int = 1, n_rep: int = 3) -> tuple[float, float]:
    """Devuelve (media, desviacion tipica) en segundos."""
    for _ in range(n_warmup):
        fn()
    sync(device)
    ts = []
    for _ in range(n_rep):
        t0 = time.perf_counter()
        fn()
        sync(device)
        ts.append(time.perf_counter() - t0)
    return float(np.mean(ts)), float(np.std(ts))


# ---------------------------------------------------------------------------

def load_picmus(phantom: str = "sim_resolution"):
    from schiffner_das import load_picmus_hdf5
    return load_picmus_hdf5(DATASETS / "PICMUS" / "database" / PHANTOM_RF[phantom])


def eval_grid(ds):
    """Rejilla canonica 612x388 — replica exacta de evaluate_fieldii._build_grid."""
    c, fs = float(ds.c_avg), float(ds.f_s)
    ns = int(ds.rf_for_angle(0).shape[0])
    K, pitch = int(ds.n_elements), float(ds.element_pitch)
    z_max = (ns - 1) * c / (2.0 * fs)
    gz = torch.linspace(5e-3, z_max, NZ, dtype=torch.float64)
    gx = torch.linspace(-K * pitch / 2.0, +K * pitch / 2.0, NX, dtype=torch.float64)
    return gz, gx


def bench_fourier_kernel(ds, gz, gx, device: str, n_angles: int, fnum_kind: str,
                         precision: str = "float64"):
    """Cronometra ``das_pw`` para ``n_angles`` angulos repartidos por el barrido.

    ``precision`` importa mucho y hay que declararla al reportar:
    ``gen_classical_picmus_fixed.py`` NO pasa el argumento, asi que los npys
    canonicos se generaron con el defecto ``float64``, mientras que ``OpHFdf``
    (el adjunto que alimenta a UNet-AG) usa ``complex64``. En una TITAN RTX el
    FP64 va a 1/32 de rendimiento, asi que comparar los segundos de los dos tal
    cual mide la precision, no el algoritmo. Se mide en las dos.
    """
    from schiffner_das import ConstantFNumber, GratingAngleLB, das_pw

    fnum = ConstantFNumber(0.0) if fnum_kind == "const0" else GratingAngleLB(60.0, 1.5)
    angles = ds.steering_angles_rad
    idx = np.linspace(0, len(angles) - 1, n_angles).round().astype(int)
    index_t0 = -int(round(ds.initial_time_s * ds.f_s))

    def make_run(aidx: int):
        rf = ds.rf_for_angle(aidx)
        ang = float(angles[aidx])

        def run():
            # das_pw imprime progreso con \r; se silencia para no ensuciar el log
            with contextlib.redirect_stdout(io.StringIO()):
                das_pw(gx, gz, rf, ds.f_s, ang,
                       ds.element_width, ds.element_pitch, ds.c_avg,
                       f_bounds=(ds.f_lb, ds.f_ub), index_t0=index_t0,
                       F_number=fnum, device=device, precision=precision,
                       tx_sign=+1.0, tx_reference="origin")
        return run

    # Un unico warmup (compilacion de kernels, contexto CUDA, cache de la FFT):
    # el kernel es identico para todos los angulos, asi que calentar en cada uno
    # multiplicaria por 3 el coste del benchmark sin cambiar el resultado.
    make_run(int(idx[0]))()
    sync(device)

    per_angle = []
    for aidx in idx:
        m, _ = timeit(make_run(int(aidx)), device, n_warmup=0, n_rep=1)
        per_angle.append(m)
        print(f"    angulo {int(aidx):3d} ({math.degrees(angles[int(aidx)]):+5.1f}°): "
              f"{m:.3f} s", flush=True)

    mu, sd = float(np.mean(per_angle)), float(np.std(per_angle))
    return dict(per_angle_s=mu, per_angle_std_s=sd, n_medidos=int(n_angles),
                precision=precision,
                total_1pw_s=mu, total_75pw_s=mu * N_ANGLES_TOTAL)


def bench_unet_ag(phantom: str, device: str):
    """UNet-AG tal como se ejecuta: adjunto Fd-F, envolvente log y red.

    Usa ``unet_ag.inference.UNetAG`` con los pesos de la semilla 42: el adjunto
    es el operador de ``make_op_for_angle`` (banda 0.5-1.5 f_c, la que procesa
    el modelo), la envolvente es ``env_log`` (CPU) y la red recibe la entrada y
    el mapa F_map reales. Se cronometran los tres componentes por separado.
    """
    from unet_ag.inference import UNetAG, load_picmus_rf
    from unet_ag.physics.f_number_map import compute_f_number_map_band_avg
    from unet_ag.physics.h_fdf_wrapper import make_op_for_angle
    from unet_ag.train.b1_attention_gate import _build_grid, _envelope_log_compress

    modelo = UNetAG.from_pretrained("s42", device=device)
    rf, geom = load_picmus_rf(DATASETS / "PICMUS" / "database" / PHANTOM_RF[phantom])
    gz, gx = _build_grid(geom.as_dict(), NZ, NX)
    op = make_op_for_angle(0.0, geom.as_dict(), gz, gx, device=device, nf_chunk=None)
    y = UNetAG._prepare_rf(rf, geom).to(device).to(op.float_dtype)

    with torch.no_grad():
        m_adj, s_adj = timeit(lambda: op.adjoint(y), device, n_warmup=1, n_rep=5)
        x0c = op.adjoint(y)
    x0_real = (x0c.real if x0c.is_complex() else x0c).float().cpu()
    m_env, s_env = timeit(lambda: _envelope_log_compress(x0_real), "cpu", n_warmup=1, n_rep=5)
    x0 = _envelope_log_compress(x0_real).to(device)[None, None]
    f_map = torch.from_numpy(compute_f_number_map_band_avg(
        gz.numpy(), gx.numpy(), geom.as_dict())).to(device)[None, None]
    with torch.no_grad():
        m_net, s_net = timeit(lambda: modelo.model(x0, f_map), device, n_warmup=3, n_rep=10)

    return dict(n_frecuencias_adjunto=int(op.Nf),
                adjunto_s=m_adj, adjunto_std_s=s_adj,
                envolvente_s=m_env, envolvente_std_s=s_env,
                red_s=m_net, red_std_s=s_net, total_s=m_adj + m_env + m_net)


def bench_drus(phantom: str, device: str, artifacts: Path):
    """DRUS en su configuración nativa (paquete ``drus``): prior en float16 en
    GPU, productos espectrales con ``Vᵀ`` en CPU, 50 pasos DDRM.

    Se reportan la carga de ``Vᵀ`` (puesta en marcha, no inferencia), el tiempo
    de extremo a extremo de una reconstrucción (media de dos, tras una de
    calentamiento) y, por separado, el de los dos componentes de un paso: la
    UNet del prior y un producto por ``Vᵀ``.
    """
    from drus.inference import DRUS

    key = {"sim_resolution": "simu_reso", "sim_contrast": "simu_cont",
           "exp_resolution": "expe_reso", "exp_contrast": "expe_cont"}[phantom]
    modelo = DRUS.from_artifacts(artifacts, device=device)
    t0 = time.perf_counter()
    ops = modelo.spectral
    t_load = time.perf_counter() - t0
    net = modelo.model

    f = DATASETS / "PICMUS" / "database" / PHANTOM_RF[phantom]
    modelo.reconstruct_picmus(f, phantom=key)          # calentamiento, no se cronometra
    sync(device)
    e2e = []
    for _ in range(2):
        t0 = time.perf_counter()
        modelo.reconstruct_picmus(f, phantom=key)
        sync(device)
        e2e.append(time.perf_counter() - t0)

    x = torch.randn(1, DRUS_CHANNELS, DRUS_SIDE, DRUS_SIDE, device=device)
    t = torch.zeros(1, device=device)
    with torch.no_grad():
        m_unet, s_unet = timeit(lambda: net(x, t), device, n_warmup=2, n_rep=5)
    vec = torch.randn(1, DRUS_CHANNELS * ops.n)
    m_vt, s_vt = timeit(lambda: ops.Vt(vec), "cpu", n_warmup=1, n_rep=2)
    return dict(carga_svd_s=t_load, extremo_a_extremo_s=float(np.mean(e2e)),
                extremo_a_extremo_runs_s=e2e, unet_s=m_unet, unet_std_s=s_unet,
                vvt_s=m_vt, vvt_std_s=s_vt,
                fraccion_espectral_por_paso=3 * m_vt / (3 * m_vt + m_unet))


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n_angles", type=int, default=8,
                    help="angulos reales a medir antes de extrapolar x75")
    ap.add_argument("--phantom", default="exp_resolution", choices=list(PHANTOM_RF),
                    help="fantoma cuyo RF se cronometra (por omision, el escenario "
                         "experimental, el de la hipotesis)")
    ap.add_argument("--drus_artifacts", type=Path, default=DRUS_ARTIFACTS)
    ap.add_argument("--precision", default="float32",
                    choices=["float64", "float32"],
                    help="precision de das_pw; float32 iguala la del adjunto (la de la tesis)")
    ap.add_argument("--skip_drus", action="store_true")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "evaluacion/metricas/tiempo_inferencia_fp32.json")
    args = ap.parse_args()

    dev = args.device
    if dev == "cuda" and not torch.cuda.is_available():
        print("CUDA no disponible; abortando (los tiempos en CPU no son comparables)")
        sys.exit(1)
    gpu = torch.cuda.get_device_name(0) if dev == "cuda" else "CPU"
    print(f"=== Tiempo de inferencia — {gpu} | das_pw precision={args.precision} ===\n")

    ds = load_picmus(args.phantom)
    gz, gx = eval_grid(ds)
    import socket
    print(f"  nodo {socket.gethostname()} | fantoma {args.phantom}", flush=True)

    print("  [1/4] DAS (ConstantFNumber(0)) ...", flush=True)
    das = bench_fourier_kernel(ds, gz, gx, dev, args.n_angles, "const0", args.precision)
    print("  [2/4] Schiffner Fd-F (GratingAngleLB(60,1.5)) ...", flush=True)
    sch = bench_fourier_kernel(ds, gz, gx, dev, args.n_angles, "gratinglb", args.precision)
    print("  [3/4] UNet-AG ...", flush=True)
    ag = bench_unet_ag(args.phantom, dev)
    drus = None
    if not args.skip_drus:
        print("  [4/4] DRUS ...", flush=True)
        try:
            drus = bench_drus(args.phantom, dev, args.drus_artifacts)
        except Exception as e:                                  # noqa: BLE001
            print(f"    DRUS fallo: {type(e).__name__}: {e}", flush=True)

    rows = [
        ("DAS 1 PW", das["total_1pw_s"]),
        # Fd-F sobre un solo angulo: el kernel ya se mide por angulo, asi que
        # este numero no requiere una medicion aparte.
        ("Schiffner Fd-F 1 PW", sch["total_1pw_s"]),
        ("DAS 75 PW", das["total_75pw_s"]),
        ("Schiffner Fd-F 75 PW", sch["total_75pw_s"]),
        ("DRUS (DDRM 50 pasos)", drus["extremo_a_extremo_s"] if drus else float("nan")),
        ("UNet-AG (propuesto)", ag["total_s"]),
    ]
    print(f"\n=== Tiempo por frame — {gpu} ===")
    hdr = f"{'metodo':24s} {'segundos':>12s} {'x DAS 1PW':>11s} {'frames/s':>10s}"
    print(hdr); print("-" * len(hdr))
    base = rows[0][1]
    for name, t in rows:
        fps = 1.0 / t if t == t and t > 0 else float("nan")
        print(f"{name:24s} {t:12.3f} {t/base:11.1f} {fps:10.3f}")

    payload = dict(gpu=gpu, nodo=socket.gethostname(),
                   precision_das_pw=args.precision, fantoma=args.phantom,
                   rejilla=[NZ, NX], das=das, schiffner=sch,
                   unet_ag=ag, drus=drus,
                   tabla={n: t for n, t in rows})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

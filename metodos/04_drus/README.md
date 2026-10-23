# DRUS — el método de difusión comparado

Réplica en Python de DRUS (Zhang et al., 2023): restauración DDRM de 50 pasos
con un prior de difusión, sobre el modelo físico explícito de PICMUS. Es el
método generativo frente al que se compara el propuesto: el más fuerte del
estudio en calidad de imagen y el más
caro en costo (115 875 GFLOP por imagen, 552.8 M parámetros).

## Condiciones de la tesis

| | |
|---|---|
| Prior | `256x256_diffusion_uncond.pt` de guided-diffusion (OpenAI), **sin ajuste fino** (la variante de mejor desempeño de la réplica) |
| Modelo físico | H y conformador apodizado B de PICMUS (Tukey 25 %, f# 1.4), rejilla 256 × 256 con x ∈ [−18, 18] mm y z ∈ [9, 48] mm (extendida respecto de la del artículo, z ∈ [10, 46] mm, para que entren los 7 alambres experimentales) |
| Descomposición | SVD densa de BH (65 536²): una factorización de 8.4 h que ocupa 51 GB junto con la matriz BH de la que parte; los factores que se conservan (U, S, Vᵀ) ocupan 34 GB |
| Muestreo | DDRM, 50 pasos, η = 0.85, η_B = 1.0, semilla 0 |
| γ por fantoma | 75 (sim. resolución), 50 (sim. contraste), 75 (exp. resolución), 25 (exp. contraste): búsqueda en la rejilla {20, 25, 50, 75}, como en el artículo original |
| Evaluación | \|x\| → 60 dB → [0, 1] → interpolación bilineal a la rejilla común de 612 × 388 |

## Uso como paquete

Se instala con el resto del repositorio (`pip install -e ".[drus]"`) y expone la
orden `drus` y la clase `drus.inference.DRUS`, con la misma forma que el
método propuesto (`unetag`, `unet_ag.inference.UNetAG`).

### 1. Artefactos (una sola vez)

```bash
export DRUS_ARTIFACTS=/ruta/a/artefactos_drus   # o --artifacts en cada orden
drus prepare operador --picmus /ruta/a/PICMUS   # minutos, CPU
drus prepare svd                                # ~8.4 h, ~120 GB de RAM, MKL ILP64 (pip install mkl)
drus prepare prior                              # descarga y verifica 2.2 GB
drus info                                       # estado y parámetros de la tesis
```

Estructura resultante:

```
$DRUS_ARTIFACTS/
├── operador/   H.npz, B.npz, grid.npz, meta.json, By_<fantoma>.npy
├── svd/        S.npy, U.npy, Vt.npy, Uty_<fantoma>.npy, svd_meta.json
└── prior/      256x256_diffusion_uncond.pt
```

En un clúster conviene separar las dos etapas de la SVD (`--stage bh` en GPU,
`--stage svd` en un proceso nuevo con toda la RAM); `slurm/svd_bh.sbatch` lo hace.

### 2. Reconstrucción

```bash
drus reconstruct --picmus contrast_speckle_expe_dataset_rf.hdf5 --out drus.png
drus reconstruct --picmus resolution_distorsion_simu_dataset_rf.hdf5 \
                 --gamma 50 --seed 1 --out drus.npy --save-x x.npy
```

```python
from drus.inference import DRUS, to_grid

modelo = DRUS.from_artifacts()
res = modelo.reconstruct_picmus("contrast_speckle_expe_dataset_rf.hdf5")
res.bmode            # (256, 256) en [0, 1]; res.x es la salida con signo
res.gamma, res.seed  # 25.0 y 0: los de la tesis para este fantoma
```

El fantoma se deduce del nombre del fichero y fija la TGC, la escala de
normalización y el γ de la tesis. Para RF propio,
`DRUS.reconstruct(rf, gamma=..., tgc_alpha=..., norm_scale=...)`. Requiere GPU
para la UNet y ~20 GB de RAM para `Vᵀ` (sus productos corren en CPU, como en
el código original). Tarda 168.0 s por imagen de extremo a extremo en
una TITAN RTX, más ~105 s de arranque para cargar los 17.2 GB de `Vᵀ`.

La reconstrucción de los cuatro fantomas para la evaluación de la tesis es
`evaluacion/scripts/gen_drus_picmus.py`.

## Contenido

| Ruta | Qué es |
|---|---|
| `codigo/drus/inference.py`, `cli.py`, `preparacion.py`, `prior.py` | API pública, orden `drus`, construcción de artefactos y prior |
| `codigo/drus/physics_picmus.py`, `observations.py` | H y B (port de `compute_svdBH.m`), observaciones `By` (port de `compute_By.m`) |
| `codigo/drus/svd_bh.py` | SVD densa vía MKL ILP64 (enlaces ctypes) |
| `codigo/drus/ddrm_sampler.py` | Muestreador DDRM (port de DRUS-v1) |
| `codigo/drus/tests/` | Pruebas del contrato de la API sobre artefactos sintéticos |
| `codigo/guided_diffusion/` | UNet del prior (de openai/guided-diffusion, sin MPI) |
| `scripts/` | Scripts por fase con los que se desarrolló la réplica (operador, SVD, ajuste fino del prior, muestreo y evaluación) |
| `slurm/` | Trabajos SLURM de la SVD y del muestreo |

El ajuste fino del prior con las 287 imágenes de supervisión del método
propuesto (`scripts/finetune_prior.py`, `codigo/drus/ft_dataset.py`) forma parte
de la réplica, pero no de la configuración evaluada en la tesis.

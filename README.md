# Reconstrucción de imágenes de ultrasonido 2D en modo B desde una sola onda plana

Repositorio de la tesis de maestría de **Héctor David De Lara Berumen**
(CIMAT, 2026): código, modelos entrenados, protocolo de evaluación y material
de reproducción de los resultados reportados.

> **Método propuesto — UNet-AG** (*B1-v5_anchor05*). Red UNet con compuertas de
> atención condicionadas por un mapa de profundidad derivado del F-number
> dependiente de la frecuencia (Fd-F) de Schiffner, que refina de forma residual
> el adjunto con apodización Fd-F de **una sola onda plana**. Sobre los
> escenarios experimentales de PICMUS y respecto del DAS de una onda plana,
> mejora la detectabilidad de lesión (gCNR) un **36.6 %** (0.528 → 0.721) y
> además mejora el contraste (CNR, −3.974 → −3.692 dB) y la resolución lateral
> (FWHM, 3.728 → 2.765 mm), en cada una de las tres semillas por separado, con
> **749 GFLOP** por imagen: 59.7 veces menos que el *beamformer* Fd-F de 75 ondas
> planas y 154.6 veces menos que el método de difusión comparado (DRUS).

---

## Estructura

| Carpeta | Contenido |
|---|---|
| `codigo/` | Los paquetes Python: `unet_ag` (método propuesto), `schiffner_das` (port del Fd-F), `drus` (réplica de DRUS) y `guided_diffusion` (red del prior de DRUS) |
| `metodos/` | Un directorio por método comparado, con sus scripts, trabajos SLURM y resultados ligeros |
| `datos/` | Generación del conjunto Field II, obtención de PICMUS y la supervisión con la que se entrenó el modelo (287 objetivos Schiffner) |
| `evaluacion/` | Protocolo de evaluación: scripts, métricas (CSV/JSON), arreglos de la rejilla de evaluación, validación MATLAB y verificaciones |
| `metodos_descartados/` | Código de los intentos que no prosperaron |
| `pyproject.toml` | Paquete instalable con dos órdenes: `unetag` (método propuesto) y `drus` (método de difusión) |

### Métodos comparados

GFLOP por imagen sobre la rejilla de 612 × 388 puntos en los escenarios
experimentales, sobre los que se enuncia la hipótesis; los tres escenarios
están en
[`evaluacion/metricas/coste_por_escenario.csv`](evaluacion/metricas/coste_por_escenario.csv).

| Directorio | Método | Entrada | GFLOP |
|---|---|---|---|
| `metodos/01_das_1pw` | DAS, apertura completa | 1 PW | 596 |
| `metodos/02_schiffner_fdf` | Schiffner Fd-F (1 y 75 PW) | 1 / 75 PW | 596 / 44 700 |
| `metodos/03_das_75pw` | DAS compuesto coherente | 75 PW | 44 700 |
| `metodos/04_drus` | DRUS (difusión, DDRM) | 1 PW | 115 875 |
| `metodos/05_unet_ag_propuesto` | **UNet-AG (propuesto)** | 1 PW | **749** |

---

## Arranque rápido

```bash
git clone git@github.com:hectorddlb/UNet-AG_Tesis.git
cd UNet-AG_Tesis
python -m venv .venv && source .venv/bin/activate
pip install -e ".[todo]"           # instala los paquetes y las órdenes `unetag` y `drus`
```

`pip install -e .` deja importables `unet_ag`, `schiffner_das` y `drus` desde
cualquier directorio. Si prefieres no instalar nada, `source env.sh` pone
`codigo/` en el `PYTHONPATH` y `pip install -r requirements.txt` cubre las
dependencias.

> **Instalación en máquinas sin internet** (nodos de cómputo, entornos aislados).
> Por omisión `pip` construye el paquete en un entorno aislado y descarga
> `setuptools` para ello, lo que falla sin red. Si ya tienes `setuptools` en el
> entorno, evita ese paso:
>
> ```bash
> pip install --no-build-isolation --no-deps -e .
> ```

### Usar el modelo propuesto

```bash
unetag info                  # comprueba que carga (3 529 366 parámetros)

# reconstruir un phantom de PICMUS
unetag reconstruct --picmus resolution_distorsion_expe_dataset_rf.hdf5 \
                   --out bmode.png --save-adjoint inicial.png

# reconstruir RF propio
unetag reconstruct --rf mis_datos.npy --geometry mi_geometria.json --out bmode.png
```

Desde Python:

```python
from unet_ag.inference import UNetAG, Geometry

modelo = UNetAG.from_pretrained()                    # semilla 42
bmode  = modelo.reconstruct_picmus("...expe_dataset_rf.hdf5")   # (612, 388) en [0,1]

# o con RF propio
geom  = Geometry(n_elements=128, n_samples=1527, pitch_m=300e-6,
                 sampling_freq_hz=20.832e6, speed_of_sound=1540.0)
bmode = modelo.reconstruct(rf, geom)                 # rf: (n_samples, n_elements)
```

Los pesos (41 MB cada uno) viven en `metodos/05_unet_ag_propuesto/checkpoints/`
y se localizan solos desde un clon; fuera de él, indica la carpeta con
`UNETAG_CHECKPOINTS=/ruta/a/checkpoints` o con `--ckpt`.

### Usar el método de difusión (DRUS)

La réplica de DRUS se encapsula igual que el método propuesto, con la orden
`drus` y la clase `drus.inference.DRUS`. Sus artefactos no caben en el
repositorio y se construyen una vez (detalles en
[`metodos/04_drus/README.md`](metodos/04_drus/README.md)):

```bash
export DRUS_ARTIFACTS=/ruta/a/artefactos_drus
drus prepare operador --picmus /ruta/a/PICMUS   # H, B y observaciones (minutos, CPU)
drus prepare svd                                # SVD densa de BH (~8.4 h, ~120 GB de RAM, MKL)
drus prepare prior                              # prior publicado por OpenAI (2.2 GB)
drus info

drus reconstruct --picmus contrast_speckle_expe_dataset_rf.hdf5 --out drus.png
```

```python
from drus.inference import DRUS

modelo = DRUS.from_artifacts()                       # $DRUS_ARTIFACTS
res = modelo.reconstruct_picmus("...expe_dataset_rf.hdf5")   # γ de la tesis, semilla 0
res.bmode                                            # (256, 256) en [0,1]
```

Requiere GPU para la UNet de 552.8 M parámetros y ~20 GB de RAM para la matriz
`Vᵀ` de la descomposición espectral.

---

## Reproducir

### La evaluación completa

Requiere PICMUS (ver [`datos/README.md`](datos/README.md)) y, para DRUS, sus
artefactos. Desde la raíz del repositorio:

```bash
export TESIS_DATASETS=/ruta/a/tus/datasets       # con PICMUS/ dentro
export DRUS_ARTIFACTS=/ruta/a/artefactos_drus
python evaluacion/scripts/gen_classical_picmus_fixed.py --device cuda   # DAS y Fd-F, 1 y 75 PW
python evaluacion/scripts/gen_b1v5_picmus_seeds.py --device cuda        # UNet-AG, 3 semillas
python evaluacion/scripts/gen_drus_picmus.py                            # DRUS, γ de la tesis
python evaluacion/scripts/eval_and_figs_finales.py                      # métricas y tablas
```

O todo en un trabajo SLURM: `sbatch evaluacion/scripts/run_resultados_finales.sbatch`.
Los métodos de 75 ondas planas dominan el tiempo (de 1 a 3.5 h en una TITAN RTX).

### El entrenamiento del método propuesto

Requiere el conjunto Field II. Los objetivos de supervisión ya están en el repo.

```bash
python -m unet_ag.train.b1_v5_pulse_aug \
    --data_root  "$TESIS_DATASETS/field_ii_data_multiangle" \
    --target_root datos/targets_schiffner_75pw \
    --output_dir  resultados/mi_corrida \
    --phantom_type points --anchor_weight 0.5 --epochs 20 --seed 42 \
    --device cuda
```

~9 h por semilla en una NVIDIA TITAN RTX de 24 GB. Los trabajos SLURM
equivalentes están en `metodos/05_unet_ag_propuesto/slurm/`.

---

## Qué está y qué no está en el repositorio

**Incluido:** todo el código; los 3 checkpoints del modelo final (122 MB); los
287 objetivos de supervisión (262 MB); los arreglos de la rejilla de evaluación
de los seis métodos (29 MB); las imágenes de la implementación de referencia en
MATLAB para la validación del port (15 MB); todas las métricas en CSV/JSON; los
resultados de las verificaciones.

**Excluido por tamaño**, con instrucciones de reconstrucción en la carpeta
correspondiente: los datasets crudos PICMUS y Field II (~30 GB); los artefactos
de DRUS (operador, SVD de 51 GB con su matriz de partida —34 GB los factores
que se conservan— y prior de difusión de 2.2 GB, que se construyen con
`drus prepare`); el dataset CUBDL.

---

## Cita

```bibtex
@mastersthesis{DeLara2026,
  author  = {De Lara Berumen, Héctor David},
  title   = {Reconstrucción de imágenes de ultrasonido 2D en modo B desde datos
             crudos RF de una sola onda plana mediante una red UNet informada
             por el F-number dependiente de la frecuencia},
  school  = {Centro de Investigación en Matemáticas (CIMAT)},
  year    = {2026},
  address = {Zacatecas, México}
}
```

## Agradecimiento a trabajos previos

Este trabajo reimplementa, adapta o compara con: el F-number dependiente de la
frecuencia de **Schiffner (2021)**; el esquema autosupervisado de **Zhang et al.
(2021)**; el método DRUS de **Zhang et al. (2023)**; las compuertas de atención
de **Oktay et al. (2018)**; el modelo de difusión de **guided-diffusion**
(OpenAI); el simulador **Field II** y el benchmark **PICMUS**.

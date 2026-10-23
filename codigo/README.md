# Código

Cuatro paquetes Python. `pip install -e .` desde la raíz los instala (con las
órdenes `unetag` y `drus`); `source env.sh` los pone en `PYTHONPATH` sin instalar.

## `unet_ag/` — el método propuesto

| Subpaquete | Contenido |
|---|---|
| `data/` | Lectores de RF (Field II, PICMUS, CUBDL), particiones y augmentación de pulso |
| `physics/` | Operador de medida `OpHFdf`, su envoltorio, el mapa de F-number y los conformadores clásicos DAS |
| `models/` | UNet con compuertas de atención condicionadas y el envoltorio residual con inicialización en cero |
| `losses/` | Pérdida compuesta MSE + L1 + SSIM |
| `metrics/` | CNR, gCNR y FWHM por alambre |
| `train/` | Puntos de entrada del linaje B1 (el final es `b1_v5_pulse_aug`) |
| `eval/` | Evaluadores sobre PICMUS, Field II y CUBDL, con sus máscaras |
| `tests/` | Pruebas de humo y de especificación |

Puntos de entrada:

```bash
unetag --help                                      # inferencia (unet_ag.inference.UNetAG)
python -m unet_ag.train.b1_v5_pulse_aug --help     # entrenamiento (método final)
python -m unet_ag.eval.evaluate_picmus  --help     # evaluador por corrida sobre PICMUS
python -m unet_ag.eval.evaluate_fieldii --help     # evaluación en dominio
```

`models/b1_v2_softdas.py` y `models/a2_unet_fourier.py` corresponden a variantes
descartadas, pero **son dependencias en tiempo de ejecución** de
`eval/evaluate_fieldii.py`, que sabe cargar cualquier checkpoint del linaje. Por
eso están aquí y no en `metodos_descartados/`.

## `schiffner_das/` — port del F-number dependiente de la frecuencia

Implementación en PyTorch del conformador en dominio de Fourier de Schiffner
(2021). Cumple dos papeles: genera los objetivos de supervisión y es método
comparado por sí mismo (1 y 75 ondas planas).

**Nota:** `das_pw()` acepta `tx_sign` y `tx_reference`. Los valores por defecto
(`-1.0`, `"edge"`) reproducen la convención del dataset original de Schiffner;
**para PICMUS y Field II hay que pasar `tx_sign=+1.0, tx_reference="origin"`**
(la corrección de la convención de transmisión).
`evaluacion/validacion_matlab/` contiene la validación numérica contra la
implementación de referencia en MATLAB.

## `drus/` — réplica del método de difusión

Encapsulada como el método propuesto: API pública en `drus.inference` (clase
`DRUS`, con `from_artifacts()`, `reconstruct()` y `reconstruct_picmus()`) y la
orden `drus` (`info`, `prepare`, `reconstruct`).

| Módulo | Contenido |
|---|---|
| `inference.py`, `cli.py` | API y orden `drus`; γ de la tesis por fantoma (`THESIS_GAMMA`) |
| `preparacion.py` | Localización (`$DRUS_ARTIFACTS`) y construcción de los artefactos: operador, SVD y prior |
| `prior.py` | Configuración y carga del prior de difusión (única fuente de sus *flags*) |
| `physics_picmus.py`, `observations.py` | Modelo físico H, conformador B y observaciones `By` |
| `svd_bh.py` | SVD densa vía MKL ILP64 |
| `ddrm_sampler.py` | Muestreador DDRM |
| `physics_fieldii.py`, `ft_dataset.py` | Variante sobre Field II y conjunto de ajuste fino del prior (fuera de la configuración evaluada) |
| `tests/` | Pruebas del contrato de la API |

Los scripts por fase y los trabajos SLURM están en `metodos/04_drus/`.

## `guided_diffusion/` — red del prior de DRUS

Copia de [openai/guided-diffusion](https://github.com/openai/guided-diffusion)
(vía DRUS-v1), sin MPI: la UNet de 256 × 256 que usa el prior.

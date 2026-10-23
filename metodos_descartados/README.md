# Métodos y variantes descartados

Código de los intentos que **no** prosperaron. Se conserva porque sobre ellos
se sustenta la conclusión de dónde conviene incorporar el conocimiento físico:
sin este código, esas afirmaciones no serían verificables.

**Nada de aquí forma parte del pipeline vigente.** El método final y los cinco
métodos comparados viven en `codigo/` y `metodos/`.

## `zhang_edn/` — propuesta inicial autosupervisada

Reimplementación de Zhang et al. (2021): red codificador-decodificador de 12
capas, construcción de la matriz de medida `H`, las cinco funciones de pérdida y
el entrenamiento en dos etapas. Su ejecución produjo el **colapso a x' ≈ 0 en
seis configuraciones**.

Nota: el único módulo de este paquete que sigue vivo es
`measurement_operator_fdf.py`, el operador `OpHFdf` — y no está aquí sino
integrado en el paquete del modelo, como `codigo/unet_ag/physics/`.

## `variantes_unet_ag/` — variantes del pivote

| Fichero | Corresponde a |
|---|---|
| `train/a1_rf_hfdf_loss.py`, `models/a1_resunet_rf.py`, `losses/fidelity_hfdf.py` | Propuesta **A1** (RF→imagen con término de fidelidad) — colapso análogo |
| `train/a2_*.py`, `losses/fourier_weighted.py`, `losses/lateral_deriv.py` | Propuesta **A2** (pérdida ponderada en Fourier), 7 variantes — `n_lat = 0/20` |
| `train/a2_v7_lateral_deriv.py` | Ablación **A2-v7** (derivada lateral), λ ∈ {0.3, 1.0, 3.0} — las tres fallan |
| `train/b1_v2_softdas.py` | **B1-v2** (frontal Soft-DAS, objetivo signado) — colapso por objetivo disperso |
| `train/b1_v4_finetune_picmus.py` | Ajuste fino sobre PICMUS, 3 modos — métricas idénticas a 4 decimales |
| `scripts/diagnose_prefilter_picmus.py` | Prefiltro en tiempo de prueba — perjudica la resolución |
| `scripts/ensemble_anchor05.py` | Ensemble de 3 semillas — ≈ la media, sin ganancia |
| `physics/classical/{cf,mv}_beamformer.py`, `scripts/precompute_{cf,mv}.py` | Líneas base CF y MV, nunca reportadas |
| `slurm/` | Los trabajos SLURM de todo lo anterior |

**Nota:** estos ficheros **no son ejecutables tal cual** desde este directorio: sus
imports asumen que forman parte del paquete `unet_ag`. Para ejecutarlos hay que
copiarlos de vuelta a `codigo/unet_ag/` en la ruta relativa que indica su
subcarpeta. Se conservan como evidencia documental, no como pipeline.

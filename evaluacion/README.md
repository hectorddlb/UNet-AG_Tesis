# Evaluación

Protocolo **único** con el que se miden los seis métodos de la tesis.
Ningún número reportado proviene de otra cadena.

```
imagen → envolvente → dB, 60 dB de rango dinámico
       → env_log normalizado a [0,1]
       → interpolación bilineal a la rejilla 612×388
       → _eval_image  (FWHM lat/ax por alambre, CNR en dB, gCNR)
```

Las figuras se dibujan de los **mismos arreglos** que alimentan las métricas
(`npys_eval_grid/`), nunca de una segunda pasada.

## Contenido

| Ruta | Qué es |
|---|---|
| `scripts/gen_classical_picmus_fixed.py` | Reconstruye DAS 1/75 PW y Fd-F 1/75 PW con la convención de transmisión corregida |
| `scripts/gen_b1v5_picmus_seeds.py` | Inferencia del modelo propuesto, 3 semillas (y el adjunto Fd-F, su entrada) |
| `scripts/gen_drus_picmus.py` | DRUS con las condiciones de la tesis (paquete `drus`) |
| `scripts/eval_and_figs_finales.py` | Aplica el protocolo y produce métricas, tablas y los arreglos de `npys_eval_grid/` |
| `scripts/run_resultados_finales.sbatch` | Las cuatro etapas anteriores en un trabajo SLURM |
| `scripts/eval_beta_sweep_canonico.py`, `eval_anchor_candidates.py` | Barrido de β y métricas del adjunto Fd-F (imagen de anclaje) |
| `scripts/coste_computacional.py` | GFLOP analíticos por escenario |
| `scripts/tiempo_inferencia.py`, `bench_coste.sbatch` | Tiempos de pared en GPU |
| `scripts/verify_*.py` | Verificaciones: convención de transmisión, Fd-F 1 PW frente al adjunto, partición y rejilla de Field II |
| `scripts/align_targets_to_train_grid.py`, `precompute_schiffner_targets_fieldii_fixed.py` | Objetivos corregidos del experimento de control |
| `metricas/` | CSV y JSON con los resultados reportados |
| `npys_eval_grid/` | `env_log` 612×388 de los 6 métodos × 4 phantoms — fuente de métricas y figuras |
| `validacion_matlab/` | Validación del port del Fd-F contra la implementación de referencia en MATLAB |
| `verificacion/` | Resultados de las verificaciones (arreglos, imágenes y JSON) |
| `control_supervision/` | CSV de los dos experimentos de control con la supervisión corregida |

## Métricas

| Fichero | Respalda |
|---|---|
| `metricas_picmus_finales.csv` | Tablas de calidad, de la hipótesis, de semillas y de las reconstrucciones de 1 PW |
| `metricas_beta_sweep.csv`, `metricas_anchor_candidates.csv` | Tabla y figura del barrido de β (con la fila del adjunto) |
| `coste_computacional.json`, `coste_por_escenario.csv` | Tablas de costo (escenario experimental y los tres escenarios) |
| `tiempo_inferencia_fp32.json` | Tiempos de la tabla de costo: los seis métodos en el escenario experimental, mismo nodo, fp32 |
| `tiempo_inferencia_fp32_carga_en_frio.json` | Corrida previa en el mismo nodo con la carga de `Vᵀ` de DRUS en frío (~105 s) |
| `metricas_fidelidad.csv` | Tabla de fidelidad (PSNR y SSIM frente al DAS 75 PW) |
| `grids_picmus.json` | Rejillas de evaluación, para redibujar sin los datasets |

`tabla_comparativa.md` es el resumen que genera `eval_and_figs_finales.py`; la
referencia son los CSV.

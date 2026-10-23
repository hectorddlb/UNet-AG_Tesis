# Métodos comparados

Un directorio por método. El código de biblioteca vive en `codigo/`; aquí están
los scripts de ejecución, los trabajos SLURM y los resultados ligeros.

| Dir | Método | Entrada | GFLOP | Rol en la tesis |
|---|---|---|---|---|
| `01_das_1pw` | DAS, apertura completa | 1 PW | 596 | Referencia de la hipótesis |
| `02_schiffner_fdf` | Schiffner Fd-F | 1 y 75 PW | 596 / 44 700 | Genera la supervisión; línea base sin parámetros (1 PW) y método comparado (75 PW) |
| `03_das_75pw` | DAS compuesto coherente | 75 PW | 44 700 | Referencia de alta calidad |
| `04_drus` | DRUS (difusión DDRM) | 1 PW | 115 875 | Método generativo comparado |
| `05_unet_ag_propuesto` | **UNet-AG** | 1 PW | **749** | La propuesta |

GFLOP por imagen en los escenarios experimentales (sobre los que se enuncia la
hipótesis); los tres escenarios están en `evaluacion/metricas/coste_por_escenario.csv`.

Todos se evalúan con el protocolo único de `evaluacion/`. Los B-modes finales de
los seis (incluido el Fd-F 1 PW) están en `evaluacion/npys_eval_grid/`.

Los trabajos SLURM de cada carpeta conservan las rutas del clúster en el que se
ejecutaron; se reutilizan pasando las rutas por variables de entorno.

## `05_unet_ag_propuesto`

- `checkpoints/anchor05_{s42,s1,s2}.pt` — los tres modelos entrenados (41 MB c/u).
- `registros/<semilla>/` — curvas de entrenamiento y validación, y las
  evaluaciones por corrida sobre PICMUS, Field II y CUBDL.
- `slurm/` — los trabajos del linaje B1.
- `submit_anchor05_seeds.sh` — lanza las tres semillas.

**Nota:** los CSV de `registros/` usan el evaluador **por corrida**, cuya referencia
«DAS 1 PW» es el adjunto Fd-F, no el DAS de apertura completa del protocolo
canónico. Sirven para comparar entrenamientos entre sí; los números publicables
son los de `evaluacion/metricas/`.

## `04_drus`

Réplica completa en Python del método de Zhang et al. (2023), encapsulada como
paquete (`codigo/drus/`, orden `drus`) igual que el método propuesto.
`README.md` explica cómo construir sus artefactos (operador, SVD y prior, que no
caben en el repositorio) y cómo reconstruir con las condiciones de la tesis.

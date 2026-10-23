# Datos

Los conjuntos crudos no están en el repositorio por tamaño (~30 GB). Esta carpeta
contiene lo necesario para reconstruirlos o localizarlos, más la supervisión
derivada, que sí se incluye porque regenerarla cuesta días de cómputo.

## Estructura esperada en ejecución

Apunta `TESIS_DATASETS` a un directorio con esta forma (o coloca los datos aquí):

```
$TESIS_DATASETS/
├── PICMUS/
│   └── database/
│       ├── simulation/{contrast_speckle,resolution_distorsion}/*.hdf5
│       └── experiments/{contrast_speckle,resolution_distorsion}/*.hdf5
└── field_ii_data_multiangle/
    ├── points/points_XXXX.mat      (300 phantoms, 75 ángulos c/u)
    └── cysts/cysts_XXXX.mat        (300 phantoms, 75 ángulos c/u)
```

## PICMUS

Benchmark público del *Plane-wave Imaging Challenge in Medical Ultrasound*
(IEEE IUS 2016). Descarga: <https://www.creatis.insa-lyon.fr/Challenge/IEEE_IUS_2016/>

Se usan los cuatro phantoms canónicos, cada uno con 75 ángulos en ±16°:
`simulation/{resolution_distorsion, contrast_speckle}` y
`experiments/{resolution_distorsion, contrast_speckle}`. Formato HDF5 según el
estándar UFF. El lector del proyecto es `unet_ag.data.rf_loader.PICMUSLoader`.

**PICMUS se reserva íntegramente para evaluación**: no interviene en el
entrenamiento en ningún momento.

## Field II

Conjunto de entrenamiento simulado: 600 phantoms (300 de reflectores puntuales
con 20 dispersores cada uno, 300 de inclusiones anecoicas con fondo de speckle y
10 quistes), simulados con 75 ángulos por phantom.

Los scripts de generación están en `field_ii/`. Requieren
[Field II](https://www.field-ii.dk/) sobre MATLAB:

| Script | Función |
|---|---|
| `generate_training_data_multiangle.m` | Genera un phantom con sus 75 ángulos |
| `run_generation_multiangle.m` | Lanza la generación del conjunto completo |
| `generate_range_cluster.m` / `run_cluster_generation.m` | Variantes por rangos, para paralelizar en clúster |

Parámetros del transductor (replican PICMUS): `f_c` =
5.208 MHz, `f_s` = 20.832 MHz, 128 elementos con paso de 300 µm y ancho de
270 µm, altura de 5 mm con foco en elevación a 20 mm, excitación de 2.5 ciclos,
respuesta al impulso gaussiana con 67 % de ancho de banda fraccional, `c` =
1540 m/s.

## Supervisión derivada: `targets_schiffner_75pw/`

Las **287 reconstrucciones Schiffner Fd-F de 75 ondas planas** que constituyen el
objetivo de entrenamiento real del modelo final, en
`points/<id>/compounded.npy`.

Se incluyen en el repositorio (262 MB) porque regenerarlas exige el conjunto
Field II completo y días de cómputo. Para regenerarlas:
`metodos/02_schiffner_fdf/scripts/precompute_schiffner_targets_fieldii.py`.

**Nota:** **Estos targets son los que se usaron para entrenar el modelo evaluado**, y
portan defectos de convención de transmisión y de rejilla. Se conservan tal cual, sin corregir, porque son la supervisión real del modelo
reportado. La versión corregida se regenera con
`evaluacion/scripts/precompute_schiffner_targets_fieldii_fixed.py` seguido de
`align_targets_to_train_grid.py`; el experimento de control correspondiente está
en `evaluacion/control_supervision/`.

## CUBDL

Usado solo para la evaluación cruzada de dominio. No se incluye;
disponible en <https://cubdl.jhu.edu/>. El subconjunto empleado son las
adquisiciones de onda plana con geometría compatible (~L7-4/L10-5).

# Validación de la reimplementación del beamformer Fd-F

El port en Python `schiffner_das.das_pw` reproduce la
implementación de referencia `das_pw.m` del repositorio
[mschiffn/f_number](https://github.com/mschiffn/f_number) (commit `ef85e74`,
MATLAB R2025b) con diferencias del orden de la precisión de máquina.

| Caso | Error rel. L2 | Dif. máx. / pico | \|ΔgCNR\| | \|ΔCNR\| (dB) |
|---|---|---|---|---|
| Fd-F, 1 PW | 9.7 × 10⁻¹⁴ | 1.1 × 10⁻¹³ | 0 | 6.4 × 10⁻¹³ |
| DAS, 1 PW | 1.3 × 10⁻¹³ | 1.3 × 10⁻¹³ | 0 | 4.6 × 10⁻¹³ |
| Fd-F, 75 PW (convención nativa) | 1.5 × 10⁻¹³ | 1.5 × 10⁻¹³ | 0 | 5.0 × 10⁻¹² |
| DAS, 75 PW (convención nativa) | 2.2 × 10⁻¹³ | 1.4 × 10⁻¹³ | 0 | 4.8 × 10⁻¹² |

Fuente: `resultados/validacion_matlab.csv`.

## Condiciones

Escenario experimental de contraste de PICMUS; rejilla de 612 × 388 puntos,
z de 5 a 50 mm y x de ±19.2 mm; ventana Tukey de parámetro 0.2; normalización
por suma de pesos; banda de 2.25 a 6.75 MHz; índice de inicio nulo; F-number
con χ_lb = 60° y F_ub = 1.5 (o apertura completa para DAS); sin antialiasing;
float64 en ambos lados.

La implementación de referencia no admite la convención de transmisión de PICMUS
para ángulos distintos de cero, así que las dos implementaciones se comparan con
75 ondas planas en la convención **nativa** de la referencia. Eso valida la
equivalencia numérica del port, pero no corresponde a los comparadores de 75
ondas planas de la evaluación, que usan la convención corregida. Con una sola onda
plana, a 0°, ambas convenciones coinciden.

Las diferencias de gCNR y CNR se calculan con las mismas regiones de interés para
las dos implementaciones (disco interior de 0.75 r en cada inclusión; anillo de
fondo de 1.25 r a 1.75 r sin las demás inclusiones), sobre la envolvente en dB
recortada a 60 dB y llevada a [0, 1].

## Contenido

| Fichero | Qué es |
|---|---|
| `preparar_entrada.py` | Empaqueta RF, geometría y rejilla en `trabajo/exp_contrast_rf.mat` (entrada común) |
| `run_das_pw_matlab.m` | Ejecuta `das_pw.m` de la referencia para una lista de ángulos |
| `comparar_python_matlab.py` | Ejecuta el port con los mismos parámetros y escribe el CSV |
| `resultados/matlab_<caso>.npy` | Imágenes complejas de MATLAB (612 × 388), ya sumadas sobre ángulos |
| `resultados/validacion_matlab.csv` | La tabla de arriba |

## Reproducir

```bash
# 1. entrada común (necesita PICMUS en $TESIS_DATASETS/PICMUS)
python evaluacion/validacion_matlab/preparar_entrada.py

# 2. referencia MATLAB (opcional: las imágenes ya están en resultados/)
git clone https://github.com/mschiffn/f_number evaluacion/validacion_matlab/trabajo/f_number
git -C evaluacion/validacion_matlab/trabajo/f_number checkout ef85e74
cd evaluacion/validacion_matlab
matlab -batch "run_das_pw_matlab('fdf', 38, 'trabajo/matlab_fdf_1pw.mat'); run_das_pw_matlab('das', 38, 'trabajo/matlab_das_1pw.mat')"
matlab -batch "run_das_pw_matlab('fdf', 1:15, 'trabajo/matlab_fdf_75_0.mat')"   # ... 16:30, ..., 61:75
matlab -batch "run_das_pw_matlab('das', 1:15, 'trabajo/matlab_das_75_0.mat')"   # ídem
cd ../..

# 3. comparación (usa los .mat de trabajo/ si existen; si no, resultados/*.npy)
python evaluacion/validacion_matlab/comparar_python_matlab.py --device cuda
```

Cada ángulo tarda ~7.5 min en MATLAB sobre CPU; los 75 ángulos se repartieron en
cinco trabajos de 15 ángulos por método. El código de la referencia no se
incluye en este repositorio por su licencia (*citationware*); se descarga de su
repositorio original.

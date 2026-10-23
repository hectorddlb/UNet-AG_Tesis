# source env.sh  — desde la raiz del repositorio
# Hace importables:  unet_ag.*        (modelo propuesto, codigo/unet_ag)
#                    schiffner_das.*  (port Fd-F, codigo/schiffner_das)
#                    drus.*           (replica DRUS, codigo/drus)
_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${_ROOT}/codigo:${PYTHONPATH:-}"
# Ruta de los datasets crudos (PICMUS, Field II), que NO viven en el repo.
export TESIS_DATASETS="${TESIS_DATASETS:-${_ROOT}/datos}"
echo "PYTHONPATH: ${_ROOT}/codigo"
echo "TESIS_DATASETS: ${TESIS_DATASETS}"

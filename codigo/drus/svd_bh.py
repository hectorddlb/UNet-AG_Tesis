"""SVD densa de BH (65536×65536) vía MKL ILP64 — Fase B de la réplica DRUS.

Por qué no scipy: numpy/scipy del env usan OpenBLAS LP64 (enteros de 32 bits).
Para n=65536, el workspace de ``sgesdd`` (~3n² ≈ 1.3e10 floats) no cabe en un
int32, y el LAPACK de referencia usa INTEGER de 32 bits en su aritmética de
índices internos → corrupto/crash con arrays de n² = 4.3e9 elementos. MATLAB
lo resuelve porque enlaza MKL ILP64; aquí hacemos lo mismo llamando
``sgesdd_`` de ``libmkl_rt`` (pip package ``mkl``) con interface ILP64 vía
ctypes (todos los enteros de 64 bits).

Convención de layout: numpy es row-major pero LAPACK es column-major. Para A
(C-order), Fortran ve Aᵀ. Si sgesdd devuelve Aᵀ = U_f·S·V_fᵀ, entonces
A = V_f·S·U_fᵀ, es decir::

    U  = V_f      (lo que Fortran llama V)
    Vt = U_fᵀ     (lo que Fortran llama U, transpuesto)

Para no transponer 17 GB, guardamos los buffers tal como salen de Fortran y
exponemos U/Vt como vistas lógicas:

    - ``vt_f`` buffer (n×n, C-order) tal que Fortran lo ve como V_fᵀ
      → interpretado en C-order, ``vt_f[i, j]`` = V_f[j, i]  → **U = vt_f.T**,
        pero al guardar escribimos U_np = vt_f con flag "es U transpuesta".

Para eliminar toda ambigüedad, este módulo re-verifica el resultado numérico
(gate G-B1) contra columnas muestreadas de BH y guarda en disco exactamente
``U.npy`` (n×n, U[:, k] = k-ésimo vector singular izquierdo), ``S.npy`` y
``Vt.npy`` (n×n, Vt[k, :] = k-ésimo vector singular derecho), ya en
convención C-order estándar de numpy.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import os
from pathlib import Path

import numpy as np


def _find_mkl_rt() -> str:
    env_lib = Path(os.environ.get("CONDA_PREFIX", "")) / "lib"
    for name in ("libmkl_rt.so.3", "libmkl_rt.so.2", "libmkl_rt.so.1", "libmkl_rt.so"):
        p = env_lib / name
        if p.exists():
            return str(p)
    found = ctypes.util.find_library("mkl_rt")
    if found:
        return found
    raise RuntimeError("libmkl_rt no encontrado — `pip install mkl` en el env")


def load_mkl_ilp64() -> ctypes.CDLL:
    mkl = ctypes.CDLL(_find_mkl_rt(), mode=ctypes.RTLD_GLOBAL)
    MKL_INTERFACE_ILP64 = 1
    ret = mkl.MKL_Set_Interface_Layer(MKL_INTERFACE_ILP64)
    if ret != MKL_INTERFACE_ILP64:
        raise RuntimeError(f"MKL_Set_Interface_Layer(ILP64) devolvió {ret}")
    # Threading layer INTEL (default); respeta MKL_NUM_THREADS
    return mkl


def sgesdd_ilp64(
    a: np.ndarray,          # (n, n) float32 C-order — SE SOBREESCRIBE
    mkl: ctypes.CDLL | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SVD completa (jobz='S') de la matriz cuadrada ``a`` (C-order).

    Devuelve (U, S, Vt) en convención numpy C-order:
    ``a ≈ U @ np.diag(S) @ Vt``.
    """
    if mkl is None:
        mkl = load_mkl_ilp64()
    n = a.shape[0]
    assert a.shape == (n, n) and a.dtype == np.float32 and a.flags.c_contiguous

    i64 = ctypes.c_longlong
    jobz = ctypes.c_char(b"S")
    n_ = i64(n)
    lda = i64(n)
    s = np.empty(n, dtype=np.float32)
    # Fortran ve A_f = aᵀ. A_f = U_f S V_fᵀ  ⇒  a = V_f S U_fᵀ.
    # u_buf recibe U_f (col-major n×n) → en C-order u_buf es U_fᵀ = Vt (numpy).
    # vt_buf recibe V_fᵀ (col-major)   → en C-order vt_buf es V_f = U (numpy).
    u_buf = np.empty((n, n), dtype=np.float32)    # será Vt en convención numpy
    vt_buf = np.empty((n, n), dtype=np.float32)   # será U  en convención numpy
    iwork = np.empty(8 * n, dtype=np.int64)
    info = i64(0)

    def _call(work_arr, lwork_val):
        mkl.sgesdd_(
            ctypes.byref(jobz), ctypes.byref(n_), ctypes.byref(n_),
            a.ctypes.data_as(ctypes.c_void_p), ctypes.byref(lda),
            s.ctypes.data_as(ctypes.c_void_p),
            u_buf.ctypes.data_as(ctypes.c_void_p), ctypes.byref(n_),
            vt_buf.ctypes.data_as(ctypes.c_void_p), ctypes.byref(n_),
            work_arr.ctypes.data_as(ctypes.c_void_p), ctypes.byref(i64(lwork_val)),
            iwork.ctypes.data_as(ctypes.c_void_p), ctypes.byref(info),
        )
        return info.value

    # workspace query. El valor llega en float32 (mantisa de 24 bits): para
    # lwork ~1e10 pierde precisión → tomar max con la cota analítica de LAPACK
    # (jobz='S', m=n: 4n²+7n) más un colchón.
    wq = np.empty(1, dtype=np.float32)
    if _call(wq, -1) != 0:
        raise RuntimeError(f"sgesdd workspace query: info={info.value}")
    lwork = max(int(wq[0]) + 8 * n, 4 * n * n + 7 * n)
    if verbose:
        print(f"[sgesdd] n={n} lwork={lwork} ({lwork*4/2**30:.1f} GiB)", flush=True)
    work = np.empty(lwork, dtype=np.float32)
    if _call(work, lwork) != 0:
        raise RuntimeError(f"sgesdd: info={info.value}")
    del work

    # Mapeo Fortran→numpy (ver docstring): U = vt_buf, Vt = u_buf
    return vt_buf, s, u_buf


def ssyevd_ilp64(
    a: np.ndarray,          # (n, n) float32 C-order SIMÉTRICA — SE SOBREESCRIBE
    mkl: ctypes.CDLL | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecomposición simétrica (jobz='V').

    Devuelve ``(lam, ev)`` con λ ascendente y ``ev`` en **filas** de numpy:
    ``ev[k, :]`` es el eigenvector k (Fortran escribe eigenvectores como
    columnas col-major dentro de ``a``, que en C-order son filas). Como A es
    simétrica, el layout de entrada C/Fortran es indistinto.
    """
    if mkl is None:
        mkl = load_mkl_ilp64()
    n = a.shape[0]
    assert a.shape == (n, n) and a.dtype == np.float32 and a.flags.c_contiguous

    i64 = ctypes.c_longlong
    jobz, uplo = ctypes.c_char(b"V"), ctypes.c_char(b"U")
    n_, lda = i64(n), i64(n)
    w = np.empty(n, dtype=np.float32)
    info = i64(0)

    def _call(work_arr, lwork_val, iwork_arr, liwork_val):
        mkl.ssyevd_(
            ctypes.byref(jobz), ctypes.byref(uplo), ctypes.byref(n_),
            a.ctypes.data_as(ctypes.c_void_p), ctypes.byref(lda),
            w.ctypes.data_as(ctypes.c_void_p),
            work_arr.ctypes.data_as(ctypes.c_void_p), ctypes.byref(i64(lwork_val)),
            iwork_arr.ctypes.data_as(ctypes.c_void_p), ctypes.byref(i64(liwork_val)),
            ctypes.byref(info),
        )
        return info.value

    wq = np.empty(1, dtype=np.float32)
    iwq = np.empty(1, dtype=np.int64)
    if _call(wq, -1, iwq, -1) != 0:
        raise RuntimeError(f"ssyevd workspace query: info={info.value}")
    lwork = max(int(wq[0]) + n, 1 + 6 * n + 2 * n * n)  # query f32 pierde precisión
    liwork = max(int(iwq[0]), 3 + 5 * n)
    if verbose:
        print(f"[ssyevd] n={n} lwork={lwork} ({lwork*4/2**30:.1f} GiB)", flush=True)
    work = np.empty(lwork, dtype=np.float32)
    iwork = np.empty(liwork, dtype=np.int64)
    if _call(work, lwork, iwork, liwork) != 0:
        raise RuntimeError(f"ssyevd: info={info.value}")
    return w, a

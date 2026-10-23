# -*- coding: utf-8 -*-
"""Pruebas de la API pública de DRUS (drus.inference, drus.preparacion).

Protegen el contrato que ve quien instala el paquete: parámetros de la tesis,
nombres de fantoma, localización de artefactos con errores accionables y la
cadena completa de inferencia sobre artefactos sintéticos diminutos (rejilla
8 x 8 y un prior de juguete), sin la SVD de 34 GB ni el prior de 2.2 GB.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from drus.inference import DRUS, THESIS_GAMMA, env_log01, to_grid
from drus.observations import PHANTOMS, PicmusRF, phantom_from_filename
from drus.preparacion import Artifacts, resolve_artifacts
from drus.prior import PRIOR_FILENAME


# ── parámetros de la tesis ──────────────────────────────────────────────
def test_gamma_de_la_tesis_por_fantoma():
    # Capítulo 6: γ elegido por fantoma en la rejilla {20, 25, 50, 75}
    assert THESIS_GAMMA == {"simu_reso": 75.0, "simu_cont": 50.0,
                            "expe_reso": 75.0, "expe_cont": 25.0}
    assert set(THESIS_GAMMA) == set(PHANTOMS)


@pytest.mark.parametrize("name,key", [
    ("resolution_distorsion_simu_dataset_rf.hdf5", "simu_reso"),
    ("contrast_speckle_simu_dataset_rf.hdf5", "simu_cont"),
    ("resolution_distorsion_expe_dataset_rf.hdf5", "expe_reso"),
    ("contrast_speckle_expe_dataset_rf.hdf5", "expe_cont"),
])
def test_fantoma_desde_el_nombre_del_fichero(name, key):
    assert phantom_from_filename(Path("/x/y") / name) == key


def test_fantoma_desconocido_pide_indicarlo():
    with pytest.raises(ValueError, match="indícalo"):
        phantom_from_filename("mis_datos.hdf5")


# ── conversiones ────────────────────────────────────────────────────────
def test_env_log01_es_simetrica_y_normalizada():
    x = np.random.default_rng(0).normal(size=(16, 16))
    a, b = env_log01(x), env_log01(-x)
    assert np.array_equal(a, b)                      # usa |x|
    assert a.dtype == np.float32 and 0.0 <= a.min() and a.max() == pytest.approx(1.0)


def test_to_grid_rellena_con_cero_fuera_del_campo():
    img = np.ones((4, 4), np.float32)
    z = np.linspace(0, 1, 4); x = np.linspace(0, 1, 4)
    out = to_grid(img, z, x, np.linspace(-1, 2, 7), np.linspace(0, 1, 4))
    assert out[0].max() == 0.0 and out[-1].max() == 0.0 and out[3].min() == 1.0


# ── localización de artefactos ──────────────────────────────────────────
def test_sin_artefactos_el_error_dice_como_construirlos(tmp_path):
    with pytest.raises(FileNotFoundError, match="drus prepare"):
        resolve_artifacts(tmp_path / "vacio")


def test_estado_de_artefactos_parciales(tmp_path):
    art = Artifacts(tmp_path)
    (art.operador).mkdir(parents=True)
    (art.operador / "B.npz").write_bytes(b"")
    (art.operador / "meta.json").write_text("{}")
    assert art.missing() == ["svd", "prior"]


# ── cadena completa sobre artefactos sintéticos ─────────────────────────
N_SIDE, NE, K = 8, 4, 200          # K > k_min + 150: la TGC fija las primeras muestras
NPIX = N_SIDE * N_SIDE


def _fake_artifacts(root: Path) -> Artifacts:
    rng = np.random.default_rng(0)
    art = Artifacts(root)
    art.operador.mkdir(parents=True); art.svd.mkdir(parents=True)
    art.prior.parent.mkdir(parents=True)
    B = sp.random(NPIX, K * NE, density=0.3, random_state=1, format="csr",
                  dtype=np.float32)
    sp.save_npz(art.operador / "B.npz", B)
    np.savez(art.operador / "grid.npz", x_axis=np.linspace(-18e-3, 18e-3, N_SIDE),
             z_axis=np.linspace(9e-3, 48e-3, N_SIDE))
    (art.operador / "meta.json").write_text(json.dumps(
        {"K": K, "k_min": 2, "Ne": NE, "fs": 20.832e6, "c": 1540.0,
         "nx": N_SIDE, "nz": N_SIDE}))
    bh = rng.normal(size=(NPIX, NPIX)).astype(np.float32)
    u, s, vt = np.linalg.svd(bh)
    np.save(art.svd / "U.npy", u.astype(np.float32))
    np.save(art.svd / "S.npy", s.astype(np.float32))
    np.save(art.svd / "Vt.npy", vt.astype(np.float32))
    art.prior.write_bytes(b"")
    return art


class _ZeroPrior(torch.nn.Module):
    """Prior de juguete: predice ruido nulo (6 canales, como learn_sigma)."""

    def forward(self, x, t):
        return torch.zeros(x.shape[0], 6, *x.shape[2:], dtype=x.dtype, device=x.device)


def _rf(seed: int = 0) -> PicmusRF:
    y = np.random.default_rng(seed).normal(size=(K + 200, NE))
    return PicmusRF(y=y, fs=20.832e6, c=1540.0, initial_time=0.0,
                    probe_x=np.linspace(-1e-3, 1e-3, NE), angle_deg=0.0)


def _model(tmp_path: Path) -> DRUS:
    m = DRUS(_fake_artifacts(tmp_path), device="cpu", timesteps=5)
    m._model = _ZeroPrior()
    return m


def test_reconstruct_devuelve_forma_y_rango(tmp_path):
    m = _model(tmp_path)
    res = m.reconstruct(_rf(), phantom="expe_cont", seed=0)
    assert res.x.shape == res.bmode.shape == (N_SIDE, N_SIDE)
    assert res.gamma == THESIS_GAMMA["expe_cont"]
    assert 0.0 <= res.bmode.min() and res.bmode.max() == pytest.approx(1.0)
    assert np.isfinite(res.x).all()


def test_reconstruct_es_determinista_con_la_semilla(tmp_path):
    m = _model(tmp_path)
    a = m.reconstruct(_rf(), phantom="simu_reso", seed=3).x
    b = m.reconstruct(_rf(), phantom="simu_reso", seed=3).x
    assert np.array_equal(a, b)


def test_uty_precalculado_coincide_con_el_de_U(tmp_path):
    m = _model(tmp_path)
    by = m.observation(_rf(), "simu_cont")
    via_u = m.spectral_observation(by, phantom=None)          # lee U por mmap
    np.save(m.artifacts.operador / "By_simu_cont.npy", by)
    np.save(m.artifacts.svd / "Uty_simu_cont.npy", via_u + 1.0)  # marca el cacheado
    cached = m.spectral_observation(by, phantom="simu_cont")
    assert np.array_equal(cached, via_u + 1.0)                 # usó el precalculado
    other = m.observation(_rf(seed=1), "simu_cont")
    assert np.allclose(m.spectral_observation(other, "simu_cont"),
                       np.load(m.artifacts.svd / "U.npy").T @ other.astype(np.float32),
                       rtol=1e-5, atol=1e-6)                   # By distinto: recalcula


def test_datos_propios_exigen_parametros_explicitos(tmp_path):
    m = _model(tmp_path)
    with pytest.raises(ValueError, match="gamma, tgc_alpha y norm_scale"):
        m.reconstruct(_rf())
    res = m.reconstruct(_rf(), gamma=30.0, tgc_alpha=4.0, norm_scale=10.0)
    assert res.phantom is None and res.gamma == 30.0


def test_prior_se_busca_con_el_nombre_publicado(tmp_path):
    assert Artifacts(tmp_path).prior.name == PRIOR_FILENAME

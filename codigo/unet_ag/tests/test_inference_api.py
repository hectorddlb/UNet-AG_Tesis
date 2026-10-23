# -*- coding: utf-8 -*-
"""Pruebas de la API pública de inferencia (unet_ag.inference).

Son las que protegen el contrato que ve alguien que instala el paquete en otra
máquina: nombres, formas, rangos y mensajes de error accionables.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from unet_ag.inference import (DYN_RANGE_DB, Geometry, UNetAG,
                               _resolve_checkpoint, to_decibels)

GEO = Geometry(n_elements=16, n_samples=256, pitch_m=300e-6,
               sampling_freq_hz=20.832e6, speed_of_sound=1540.0)


def _tiny_model(tmp_path: Path) -> UNetAG:
    """Un UNetAG con pesos aleatorios, sin depender de los checkpoints."""
    from unet_ag.models.b1_v3_residual import B1V3Residual
    ck = tmp_path / "tiny.pt"
    torch.save(B1V3Residual(base_ch=32, embed_ch=16).state_dict(), ck)
    return UNetAG(ck, device="cpu", nz=48, nx=32)


# ── contrato de Geometry ────────────────────────────────────────────────
def test_geometry_expone_las_claves_que_espera_el_operador():
    d = GEO.as_dict()
    for k in ("n_elements", "n_samples", "pitch_m", "sampling_freq_hz",
              "speed_of_sound", "element_width_m", "fc", "t_start"):
        assert k in d, f"falta la clave {k} que consume make_op_for_angle"


def test_geometry_es_inmutable():
    with pytest.raises(Exception):
        GEO.n_elements = 999          # frozen dataclass


# ── contrato de la reconstrucción ───────────────────────────────────────
def test_reconstruct_devuelve_envolvente_normalizada(tmp_path):
    m = _tiny_model(tmp_path)
    rf = np.random.default_rng(0).normal(size=(GEO.n_samples, GEO.n_elements))
    out = m.reconstruct(rf, GEO)
    assert out.shape == (48, 32)
    assert out.dtype == np.float32
    assert 0.0 <= float(out.min()) and float(out.max()) <= 1.0


def test_reconstruct_puede_devolver_la_solucion_inicial(tmp_path):
    m = _tiny_model(tmp_path)
    rf = np.random.default_rng(1).normal(size=(GEO.n_samples, GEO.n_elements))
    bmode, adjunto = m.reconstruct(rf, GEO, return_adjoint=True)
    assert bmode.shape == adjunto.shape
    # el adjunto es una reconstruccion por si mismo, no una constante
    assert float(adjunto.std()) > 0.0


def test_rf_transpuesto_da_error_legible(tmp_path):
    m = _tiny_model(tmp_path)
    rf = np.zeros((GEO.n_elements, GEO.n_samples))      # al reves
    with pytest.raises(ValueError, match="transpuesto"):
        m.reconstruct(rf, GEO)


def test_rf_no_2d_da_error(tmp_path):
    m = _tiny_model(tmp_path)
    with pytest.raises(ValueError, match="2-D"):
        m.reconstruct(np.zeros((2, GEO.n_samples, GEO.n_elements)), GEO)


def test_acepta_tensor_de_torch_igual_que_numpy(tmp_path):
    m = _tiny_model(tmp_path)
    rf = np.random.default_rng(2).normal(size=(GEO.n_samples, GEO.n_elements))
    a = m.reconstruct(rf, GEO)
    b = m.reconstruct(torch.from_numpy(rf), GEO)
    np.testing.assert_allclose(a, b, atol=1e-5)


# ── utilidades y errores ────────────────────────────────────────────────
def test_grid_tiene_las_dimensiones_pedidas(tmp_path):
    m = _tiny_model(tmp_path)
    gz, gx = m.grid(GEO)
    assert gz.shape == (48,) and gx.shape == (32,)
    assert gz[0] < gz[-1] and gx[0] < gx[-1]


def test_to_decibels_mapea_el_rango_completo():
    np.testing.assert_allclose(to_decibels(np.array([0.0, 1.0])),
                               np.array([-DYN_RANGE_DB, 0.0]))


def test_checkpoint_inexistente_da_error_accionable(tmp_path):
    with pytest.raises(FileNotFoundError, match="No se encontró el checkpoint"):
        UNetAG(tmp_path / "no_existe.pt", device="cpu")


def test_resolver_de_checkpoints_respeta_la_variable_de_entorno(tmp_path, monkeypatch):
    d = tmp_path / "ckpts"
    d.mkdir()
    (d / "anchor05_s42.pt").write_bytes(b"")
    monkeypatch.setenv("UNETAG_CHECKPOINTS", str(d))
    assert _resolve_checkpoint("s42") == d / "anchor05_s42.pt"


def test_resolver_lista_donde_busco_al_fallar(tmp_path, monkeypatch):
    # Dentro de un clon del repo el resolvedor SI encuentra los pesos reales
    # (que es lo correcto), asi que para probar el fallo hay que simular una
    # instalacion sin ellos: se sustituyen los candidatos por rutas vacias.
    import unet_ag.inference as inf
    monkeypatch.setattr(inf, "_candidate_checkpoint_dirs",
                        lambda: [tmp_path / "vacio_a", tmp_path / "vacio_b"])
    with pytest.raises(FileNotFoundError) as e:
        inf._resolve_checkpoint("s1")
    msg = str(e.value)
    assert "anchor05_s1.pt" in msg
    assert "vacio_a" in msg and "vacio_b" in msg   # dice donde busco
    assert "UNETAG_CHECKPOINTS" in msg             # y como arreglarlo


def test_from_pretrained_rechaza_semilla_invalida():
    with pytest.raises(ValueError, match="seed debe ser uno de"):
        UNetAG.from_pretrained(seed="s99")

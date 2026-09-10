import math

import pytest

from src.utils import angulo_corpo


def test_angulo_reto_noventa_graus():
    assert angulo_corpo([1, 0], [0, 0], [0, 1]) == pytest.approx(90.0)


def test_angulo_estendido_cento_e_oitenta_graus():
    assert angulo_corpo([0, 1], [0, 0], [0, -1]) == pytest.approx(180.0)


def test_angulo_fechado_quarenta_e_cinco_graus():
    assert angulo_corpo([1, 0], [0, 0], [1, 1]) == pytest.approx(45.0)


def test_angulo_vetores_identicos_eh_zero():
    assert angulo_corpo([1, 1], [0, 0], [2, 2]) == pytest.approx(0.0, abs=1e-3)

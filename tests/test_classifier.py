from dataclasses import dataclass

from src.classifier import Classifica


@dataclass
class FakeLandmark:
    x: float
    y: float
    z: float = 0.0


@dataclass
class FakeDetectionResult:
    pose_landmarks: list


def _build_result(ombro, cotovelo, pulso):
    """Monta um resultado falso com só os 33 pontos que o classificador usa (11, 13, 15)."""
    pontos = [FakeLandmark(0, 0)] * 33
    pontos[11] = FakeLandmark(*ombro)
    pontos[13] = FakeLandmark(*cotovelo)
    pontos[15] = FakeLandmark(*pulso)
    return FakeDetectionResult(pose_landmarks=[pontos])


def test_sem_ninguem_na_camera():
    resultado = FakeDetectionResult(pose_landmarks=[])
    assert Classifica(resultado) == "Ninguém na câmera"


def test_braco_esticado_para_tras():
    resultado = _build_result(ombro=(0, 0), cotovelo=(0, 1), pulso=(0, 2))
    assert Classifica(resultado) == "Braço esticado para trás"


def test_braco_flexionado():
    resultado = _build_result(ombro=(0, 1), cotovelo=(0, 0), pulso=(1, 0))
    assert Classifica(resultado) == "Braço flexionado"


def test_braco_reto_para_baixo():
    resultado = _build_result(ombro=(1, 1), cotovelo=(0, 0), pulso=(1, 0))
    assert Classifica(resultado) == "Braço reto para baixo"

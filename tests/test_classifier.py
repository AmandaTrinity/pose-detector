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


def test_sem_ninguem_na_camera():
    resultado = FakeDetectionResult(pose_landmarks=[])
    assert Classifica(resultado) == 'Ninguém na câmera'


def test_classifica_retorna_um_sinal_valido():
    pontos = [FakeLandmark(0.5, 0.5, 0.0) for _ in range(33)]
    resultado_deteccao = FakeDetectionResult(pose_landmarks=[pontos])

    predicao = Classifica(resultado_deteccao)

    assert isinstance(predicao, str)
    assert predicao != ''

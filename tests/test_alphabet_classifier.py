from dataclasses import dataclass

from src.alphabet_classifier import Classifica


@dataclass
class FakeLandmark:
    x: float
    y: float
    z: float = 0.0


@dataclass
class FakeDetectionResult:
    hand_landmarks: list


def test_sem_mao_na_camera():
    resultado = FakeDetectionResult(hand_landmarks=[])
    assert Classifica(resultado) == 'Nenhuma mão na câmera'


def test_classifica_retorna_uma_letra_valida():
    pontos = [FakeLandmark(0.5, 0.5, 0.0) for _ in range(21)]
    resultado_deteccao = FakeDetectionResult(hand_landmarks=[pontos])

    predicao = Classifica(resultado_deteccao)

    assert isinstance(predicao, str)
    assert predicao != ''

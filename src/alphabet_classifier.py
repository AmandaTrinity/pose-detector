import joblib

from src.landmarks_utils import normalizar_landmarks

MODELO_PATH = 'models/alphabet_classifier.joblib'

_modelo = None
_label_encoder = None


def _carregar_modelo():
    global _modelo, _label_encoder
    if _modelo is None:
        dados = joblib.load(MODELO_PATH)
        _modelo = dados['modelo']
        _label_encoder = dados['label_encoder']
    return _modelo, _label_encoder


def Classifica(detection_result):
    hand_landmarks = detection_result.hand_landmarks  # pegamos a lista

    if not hand_landmarks:
        return 'Nenhuma mão na câmera'

    # inicialmente, configurado apenas para uma mão só (num_hands=1 no HandDetector)
    # mesma ordem de features usada no treino (ver dataset/alphabet_schema.py)
    coords_planas = []
    for ponto in hand_landmarks[0]:
        coords_planas.extend([ponto.x, ponto.y, ponto.z])

    # mesma normalização (origem no pulso, escala pela mão) usada na extração do dataset -
    # sem isso, o modelo veria coordenadas numa escala bem diferente da vista no treino
    landmarks = normalizar_landmarks(coords_planas)

    modelo, label_encoder = _carregar_modelo()
    predicao = modelo.predict([landmarks])[0]
    return label_encoder.inverse_transform([predicao])[0]

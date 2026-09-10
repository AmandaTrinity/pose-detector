import joblib

MODELO_PATH = 'models/pose_classifier.joblib'

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
    pose_landmarks = detection_result.pose_landmarks  # pegamos a lista

    if not pose_landmarks:
        return 'Ninguém na câmera'

    # inicialmente, configurado apenas para uma pessoa só
    # mesma ordem de features usada no treino (ver dataset/schema.py)
    landmarks = []
    for ponto in pose_landmarks[0]:
        landmarks.extend([ponto.x, ponto.y, ponto.z])

    modelo, label_encoder = _carregar_modelo()
    predicao = modelo.predict([landmarks])[0]
    return label_encoder.inverse_transform([predicao])[0]

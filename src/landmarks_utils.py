import numpy as np

# indices dentro dos 21 landmarks de mao (ver dataset/alphabet_schema.py)
WRIST_IDX = 0
ESCALA_REF_IDX = 9  # middle_finger_mcp: base do dedo medio, referencia estavel de "tamanho da mao"


def normalizar_landmarks(coords_planas):
    """Normaliza 21 landmarks de mao (63 valores x,y,z) pra ficarem invariantes a posicao e
    distancia da camera: origem no pulso, escala pela distancia pulso -> base do dedo medio.

    Sem isso, o modelo treina com imagens recortadas bem de perto (mao ocupa o frame quase
    todo) mas na webcam ao vivo a mao e so uma fracao pequena do frame, numa distancia/posicao
    variavel - as coordenadas absolutas ficam numa escala bem diferente da vista no treino.
    """
    pontos = np.asarray(coords_planas, dtype=float).reshape(21, 3)

    origem = pontos[WRIST_IDX]
    transladados = pontos - origem

    escala = np.linalg.norm(transladados[ESCALA_REF_IDX])
    if escala == 0:
        escala = 1.0  # mao degenerada (todos os pontos no mesmo lugar), evita divisao por zero

    normalizados = transladados / escala
    return normalizados.flatten().tolist()

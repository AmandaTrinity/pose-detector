from utils import angulo_corpo

def Classifica(detection_result):
    pose_landmarks = detection_result.pose_landmarks #pegamos a lista

    if not pose_landmarks:
        return 'Ninguém na câmera'
    # inicialmente, configurado apenas para uma pessoa só
    ombro = [pose_landmarks[0][11].x, pose_landmarks[0][11].y]
    cotovelo = [pose_landmarks[0][13].x, pose_landmarks[0][13].y]
    pulso = [pose_landmarks[0][15].x, pose_landmarks[0][15].y]

    # ângulo
    angulo_braço = angulo_corpo(pulso, cotovelo, ombro)
    if angulo_braço >= 170:
        return 'Braço esticado para trás'
    elif angulo_braço >= 90:
        return 'Braço flexionado'
    else:
        return 'Braço reto para baixo'
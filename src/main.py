import cv2
import time
from src.pose_detector import PoseDetector
from src.classifier import Classifica

pose_detector = PoseDetector()

cap = cv2.VideoCapture(0) # ligar webcam

while True:
    ret, frame = cap.read() # ler quadro atual
    if not ret:
        break
    timestamp_ms = int(time.time() * 1000)
    result = pose_detector.pose_detection(frame, timestamp_ms)
    annotated_image = pose_detector.draw_landmarks(frame, result)

    classificacao = Classifica(result)
    print(classificacao)

    # escreve o sinal reconhecido direto no vídeo (com fundo pra ficar legível)
    (largura_texto, altura_texto), _ = cv2.getTextSize(classificacao, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    cv2.rectangle(annotated_image, (5, 10), (15 + largura_texto, 25 + altura_texto), (0, 0, 0), -1)
    cv2.putText(
        annotated_image, classificacao, (10, 20 + altura_texto),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA
    )

    cv2.imshow('Pose Detection', annotated_image) # mostrar pose
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release() # desligar webcam
cv2.destroyAllWindows() # fechar janela

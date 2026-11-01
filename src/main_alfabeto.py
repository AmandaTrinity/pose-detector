import cv2
import time
from mediapipe.tasks.python import vision
from src.hand_detector import HandDetector
from src.alphabet_classifier import Classifica

hand_detector = HandDetector(running_mode=vision.RunningMode.VIDEO)

cap = cv2.VideoCapture(0)  # ligar webcam

while True:
    ret, frame = cap.read()  # ler quadro atual
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    timestamp_ms = int(time.time() * 1000)
    result = hand_detector.hand_detection_video(frame_rgb, timestamp_ms)
    annotated_image = hand_detector.draw_landmarks(frame, result)

    classificacao = Classifica(result)
    print(classificacao)

    # escreve a letra reconhecida direto no vídeo (com fundo pra ficar legível)
    (largura_texto, altura_texto), _ = cv2.getTextSize(classificacao, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
    cv2.rectangle(annotated_image, (5, 10), (15 + largura_texto, 25 + altura_texto), (0, 0, 0), -1)
    cv2.putText(
        annotated_image, classificacao, (10, 20 + altura_texto),
        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA
    )

    cv2.imshow('Alfabeto em Libras', annotated_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # desligar webcam
cv2.destroyAllWindows()  # fechar janela

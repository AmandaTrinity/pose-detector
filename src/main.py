import cv2
import time
from pose_detector import PoseDetector
from classifier import Classifica

pose_detector = PoseDetector()

cap = cv2.VideoCapture(0) # ligar webcam

while True:
    ret, frame = cap.read() # ler quadro atual
    if not ret:
        break
    timestamp_ms = int(time.time() * 1000)
    result = pose_detector.pose_detection(frame, timestamp_ms)
    annotated_image = pose_detector.draw_landmarks(frame, result)
    cv2.imshow('Pose Detection', annotated_image) # mostrar pose
    classificacao = Classifica(result)
    print(classificacao)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release() # desligar webcam
cv2.destroyAllWindows() # fechar janela

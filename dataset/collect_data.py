import os
import cv2
import time
from src.pose_detector import PoseDetector

pose_detector = PoseDetector()

cap = cv2.VideoCapture(0) # ligar webcam

gravar = False
contador=0
palavra_atual=''

# se apertar 'a' é a palavra 'oi' agora
# se apertar 'b' é a palavra 'obrigado' agora
# se apertar 'c' é a palavra 'eu te amo' agora
    
while True:
    ret, frame = cap.read() # ler quadro atual
    if not ret:
        break
    timestamp_ms = int(time.time() * 1000)
    result = pose_detector.pose_detection(frame, timestamp_ms)
    annotated_image = pose_detector.draw_landmarks(frame, result)
    cv2.imshow('Pose Detection', annotated_image) # mostrar pose

    # tava tendo bug quando apertava mais de 1 letra seguida, então agora vai funcionar direitinho
    tecla=cv2.waitKey(1) & 0xFF
    if tecla == ord('a'):
        contador=0
        gravar = True
        palavra_atual = 'oi'

    if tecla == ord('b'):
        contador=0
        gravar = True
        palavra_atual = 'obrigado'

    if tecla == ord('c'):
        contador=0
        gravar = True
        palavra_atual = 'eu te amo'
    if tecla == ord('q'):
        break

    if gravar == True:
        if contador < 30:
            # para treinar uma IA, o melhor jeito é usar dados reais 
            if result.pose_landmarks: # tem que ter alguém na câmera
                linha_dados=[]
                for ponto in result.pose_landmarks[0]:
                    linha_dados.append(ponto.x)
                    linha_dados.append(ponto.y)
                    linha_dados.append(ponto.z)
                
                # adicionar a label
                linha_dados.append(palavra_atual)

                with open('dataset/dataset.csv', 'a') as f:
                    f.write(','.join(map(str, linha_dados)) + '\n')
            
            contador+=1
        else:
            gravar = False

cap.release() # desligar webcam
cv2.destroyAllWindows() # fechar janela

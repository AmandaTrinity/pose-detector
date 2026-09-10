import os
import re

import cv2

from src.pose_detector import PoseDetector

VIDEOS_DIR = 'dataset/raw_videos'
OUTPUT_CSV = 'dataset/dataset.csv'
MAX_FRAMES_POR_VIDEO = 30  # mesmo limite usado na coleta manual (collect_data.py)

# ex: "01AcontecerSinalizador01-1.mp4" -> label "acontecer"
NOME_ARQUIVO_RE = re.compile(r'^\d{2}([A-Za-zÀ-ÿ]+)Sinalizador\d{2}-\d+\.mp4$')


def extrair_label(nome_arquivo):
    match = NOME_ARQUIVO_RE.match(nome_arquivo)
    if not match:
        return None
    return match.group(1).lower()


def processar_video(caminho_video, video_id, label, escritor):
    # cada vídeo ganha seu próprio PoseDetector: o modo VIDEO do MediaPipe exige
    # timestamps monotonicamente crescentes por instância, e cada vídeo reinicia a contagem do zero
    pose_detector = PoseDetector()
    cap = cv2.VideoCapture(caminho_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    frames_gravados = 0
    frame_idx = 0
    while cap.isOpened() and frames_gravados < MAX_FRAMES_POR_VIDEO:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp_ms = int((frame_idx / fps) * 1000)
        resultado = pose_detector.pose_detection(frame, timestamp_ms)

        if resultado.pose_landmarks:
            linha_dados = []
            for ponto in resultado.pose_landmarks[0]:
                linha_dados.extend([ponto.x, ponto.y, ponto.z])
            # video_id identifica o vídeo de origem (pra dar pra agrupar train/test sem vazar dado)
            linha_dados.append(video_id)
            linha_dados.append(label)
            escritor.write(','.join(map(str, linha_dados)) + '\n')
            frames_gravados += 1

        frame_idx += 1

    cap.release()
    return frames_gravados


def main():
    arquivos = sorted(f for f in os.listdir(VIDEOS_DIR) if f.endswith('.mp4'))
    print(f'{len(arquivos)} vídeos encontrados em {VIDEOS_DIR}')

    with open(OUTPUT_CSV, 'w') as escritor:
        for i, nome_arquivo in enumerate(arquivos, start=1):
            label = extrair_label(nome_arquivo)
            if label is None:
                print(f'[{i}/{len(arquivos)}] pulando (nome fora do padrão): {nome_arquivo}')
                continue

            caminho_video = os.path.join(VIDEOS_DIR, nome_arquivo)
            video_id = os.path.splitext(nome_arquivo)[0]
            frames_gravados = processar_video(caminho_video, video_id, label, escritor)
            print(f'[{i}/{len(arquivos)}] {nome_arquivo} -> label="{label}", {frames_gravados} frames gravados')

    print(f'Extração concluída. Dataset salvo em {OUTPUT_CSV}')


if __name__ == '__main__':
    main()

import multiprocessing
import os

import cv2

from src.hand_detector import HandDetector
from src.landmarks_utils import normalizar_landmarks

IMAGES_DIR = 'dataset/raw_alphabet'
OUTPUT_CSV = 'dataset/dataset_alfabeto.csv'
SPLITS = ['train', 'test']


def processar_classe(args):
    # cada processo cria seu proprio HandDetector (modo IMAGE, sem estado entre chamadas,
    # entao nao ha problema de varias instancias rodando em paralelo)
    split, letra = args
    detector = HandDetector()

    pasta = os.path.join(IMAGES_DIR, split, letra)
    arquivos = sorted(os.listdir(pasta))

    linhas = []
    for nome_arquivo in arquivos:
        frame_bgr = cv2.imread(os.path.join(pasta, nome_arquivo))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        resultado = detector.hand_detection_image(frame_rgb)

        if resultado.hand_landmarks:
            coords_planas = []
            for ponto in resultado.hand_landmarks[0]:
                coords_planas.extend([ponto.x, ponto.y, ponto.z])
            linha_dados = normalizar_landmarks(coords_planas)
            linha_dados.append(split)
            linha_dados.append(letra)
            linhas.append(','.join(map(str, linha_dados)))

    print(f'  {split}/{letra}: {len(linhas)}/{len(arquivos)} imagens com mao detectada', flush=True)
    return split, letra, len(arquivos), linhas


def main():
    tarefas = [
        (split, letra)
        for split in SPLITS
        for letra in sorted(os.listdir(os.path.join(IMAGES_DIR, split)))
    ]

    with open(OUTPUT_CSV, 'w') as escritor, multiprocessing.Pool() as pool:
        total_imagens = 0
        total_detectadas = 0
        for split, letra, n_imagens, linhas in pool.imap_unordered(processar_classe, tarefas):
            for linha in linhas:
                escritor.write(linha + '\n')
            total_imagens += n_imagens
            total_detectadas += len(linhas)

    print(f'Extracao concluida: {total_detectadas}/{total_imagens} imagens com mao detectada.')
    print(f'Dataset salvo em {OUTPUT_CSV}')


if __name__ == '__main__':
    main()

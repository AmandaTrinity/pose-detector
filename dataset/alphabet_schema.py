import pandas as pd

# ordem oficial dos 21 landmarks do MediaPipe Hands (mesma usada em src/hand_detector.py)
HAND_LANDMARK_NAMES = [
    'wrist',
    'thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip',
    'index_finger_mcp', 'index_finger_pip', 'index_finger_dip', 'index_finger_tip',
    'middle_finger_mcp', 'middle_finger_pip', 'middle_finger_dip', 'middle_finger_tip',
    'ring_finger_mcp', 'ring_finger_pip', 'ring_finger_dip', 'ring_finger_tip',
    'pinky_mcp', 'pinky_pip', 'pinky_dip', 'pinky_tip',
]

COLUMN_NAMES = [
    f'{nome}_{eixo}' for nome in HAND_LANDMARK_NAMES for eixo in ('x', 'y', 'z')
] + ['split', 'label']


def carregar_dataset(caminho='dataset/dataset_alfabeto.csv'):
    """Le o dataset_alfabeto.csv (sem header, gerado por extract_from_alphabet_images.py) com nomes de coluna."""
    return pd.read_csv(caminho, header=None, names=COLUMN_NAMES)

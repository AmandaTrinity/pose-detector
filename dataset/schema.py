import pandas as pd

# ordem oficial dos 33 landmarks do MediaPipe Pose (mesmos índices usados em src/classifier.py)
POSE_LANDMARK_NAMES = [
    'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
    'right_eye_inner', 'right_eye', 'right_eye_outer',
    'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
    'left_index', 'right_index', 'left_thumb', 'right_thumb',
    'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index',
]

COLUMN_NAMES = [
    f'{nome}_{eixo}' for nome in POSE_LANDMARK_NAMES for eixo in ('x', 'y', 'z')
] + ['video_id', 'label']


def carregar_dataset(caminho='dataset/dataset.csv'):
    """Le o dataset.csv (sem header, gerado por collect_data.py/extract_from_videos.py) com nomes de coluna."""
    return pd.read_csv(caminho, header=None, names=COLUMN_NAMES)

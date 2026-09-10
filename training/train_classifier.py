import os

import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder

from dataset.schema import carregar_dataset

DATASET_CSV = 'dataset/dataset.csv'
MODELO_SAIDA = 'models/pose_classifier.joblib'
MATRIZ_CONFUSAO_SAIDA = 'reports/confusion_matrix.png'
N_FOLDS = 5  # cada sinal tem só 4-5 vídeos (takes), então 5-fold ~ deixar 1 take de fora por vez


def avaliar_com_cv(modelo, X, y, grupos, n_folds=N_FOLDS):
    """Predições out-of-fold via StratifiedGroupKFold: nenhum frame do mesmo vídeo aparece em
    treino e teste ao mesmo tempo (evita vazamento entre frames quase idênticos de um mesmo take),
    e a proporção de classes é mantida em cada fold."""
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    return cross_val_predict(modelo, X, y, cv=cv, groups=grupos)


def main():
    df = carregar_dataset(DATASET_CSV)
    X = df.drop(columns=['label', 'video_id']).values
    y_texto = df['label'].values
    grupos = df['video_id'].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_texto)

    modelo = RandomForestClassifier(n_estimators=200, random_state=42)

    # com poucos vídeos por classe, usar todos os folds (em vez de um único split
    # treino/teste) dá uma estimativa de acurácia bem mais confiável
    y_pred = avaliar_com_cv(modelo, X, y, grupos)

    print(classification_report(y, y_pred, target_names=encoder.classes_, zero_division=0))

    os.makedirs(os.path.dirname(MATRIZ_CONFUSAO_SAIDA), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay.from_predictions(
        y, y_pred, display_labels=encoder.classes_, xticks_rotation=90, ax=ax
    )
    ax.set_title(f'Matriz de confusão (predições out-of-fold, {N_FOLDS}-fold agrupado por vídeo)')
    plt.tight_layout()
    plt.savefig(MATRIZ_CONFUSAO_SAIDA)
    print(f'Matriz de confusão salva em {MATRIZ_CONFUSAO_SAIDA}')

    # o modelo salvo é treinado com todos os dados disponíveis (deploy real),
    # as métricas acima vêm da validação cruzada, não desse modelo final
    modelo.fit(X, y)
    os.makedirs(os.path.dirname(MODELO_SAIDA), exist_ok=True)
    joblib.dump({'modelo': modelo, 'label_encoder': encoder}, MODELO_SAIDA)
    print(f'Modelo final (treinado com todos os dados) salvo em {MODELO_SAIDA}')


if __name__ == '__main__':
    main()

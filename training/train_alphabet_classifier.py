import os

import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.preprocessing import LabelEncoder

from dataset.alphabet_schema import carregar_dataset

DATASET_CSV = 'dataset/dataset_alfabeto.csv'
MODELO_SAIDA = 'models/alphabet_classifier.joblib'
MATRIZ_CONFUSAO_SAIDA = 'reports/alphabet_confusion_matrix.png'


def main():
    df = carregar_dataset(DATASET_CSV)

    df_train = df[df['split'] == 'train']
    df_test = df[df['split'] == 'test']

    X_train = df_train.drop(columns=['label', 'split']).values
    X_test = df_test.drop(columns=['label', 'split']).values

    # mesmo encoder pros dois splits, treinado com as classes do dataset inteiro
    encoder = LabelEncoder()
    encoder.fit(df['label'].values)
    y_train = encoder.transform(df_train['label'].values)
    y_test = encoder.transform(df_test['label'].values)

    modelo = RandomForestClassifier(n_estimators=200, random_state=42)
    modelo.fit(X_train, y_train)

    # dataset ja vem com split oficial de treino/teste (verificado sem leakage no EDA),
    # entao a avaliacao e feita direto no teste, sem precisar de cross-validation
    y_pred = modelo.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=encoder.classes_, zero_division=0))

    os.makedirs(os.path.dirname(MATRIZ_CONFUSAO_SAIDA), exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=encoder.classes_, xticks_rotation=90, ax=ax
    )
    ax.set_title('Matriz de confusao (split de teste oficial)')
    plt.tight_layout()
    plt.savefig(MATRIZ_CONFUSAO_SAIDA)
    print(f'Matriz de confusao salva em {MATRIZ_CONFUSAO_SAIDA}')

    # o modelo salvo e treinado com todos os dados disponiveis (deploy real, igual ao pose_classifier),
    # as metricas acima vem da avaliacao no teste, nao desse modelo final
    X_todos = df.drop(columns=['label', 'split']).values
    y_todos = encoder.transform(df['label'].values)
    modelo.fit(X_todos, y_todos)

    os.makedirs(os.path.dirname(MODELO_SAIDA), exist_ok=True)
    joblib.dump({'modelo': modelo, 'label_encoder': encoder}, MODELO_SAIDA)
    print(f'Modelo final (treinado com todos os dados) salvo em {MODELO_SAIDA}')


if __name__ == '__main__':
    main()

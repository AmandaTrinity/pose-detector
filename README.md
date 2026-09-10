# Detector de Poses e Coletor de Dados de Libras

Este projeto utiliza **Python**, **OpenCV** e a moderna **Tasks API do MediaPipe (Google)** para detectar pontos de referência do corpo humano (landmarks) em tempo real via webcam.

---

## 🏗️ Estrutura do Projeto

O projeto foi organizado de forma modular (Orientação a Objetos) para separar as responsabilidades:

*   **`models/`**: `pose_landmarker_lite.task` (modelo pré-treinado do MediaPipe, versionado) e `pose_classifier.joblib` (nosso classificador de sinais — **não versionado**, gerado rodando `training/train_classifier.py`).
*   **`dataset/`**:
    *   `collect_data.py`: grava coordenadas do corpo via webcam e salva em `dataset.csv` (cada gravação vira uma "sessão" com um `video_id` próprio).
    *   `extract_from_videos.py`: roda o `PoseDetector` sobre vídeos de `raw_videos/` (não versionado — vem do Kaggle) e gera `dataset.csv` a partir deles.
    *   `schema.py`: nomes de coluna do `dataset.csv` (33 landmarks × x/y/z + `video_id` + `label`) e função `carregar_dataset()`.
    *   `dataset.csv`: dataset atual — 2970 amostras, 20 sinais de Libras, extraídas do [MINDS-Libras](https://www.kaggle.com/datasets/j0aopsantos/minds-libras) (Sinalizador01).
*   **`notebooks/eda.ipynb`**: análise exploratória do dataset (distribuição de classes, ranges dos landmarks, projeção PCA).
*   **`training/train_classifier.py`**: treina um `RandomForestClassifier` sobre o `dataset.csv` e salva o modelo + relatório de métricas.
*   **`reports/`**: saídas do treino (matriz de confusão) — **não versionado**, regenerado a cada treino.
*   **`src/`**:
    *   `main.py`: O "maestro" do projeto. Liga a webcam, coordena as classes e mostra o resultado na tela.
    *   `pose_detector.py`: Classe *Wrapper* que encapsula e facilita o uso da IA do MediaPipe.
    *   `utils.py`: Funções matemáticas de suporte (`angulo_corpo` — não usada mais pelo `classifier.py` atual, mas mantida com testes).
    *   `classifier.py`: carrega `models/pose_classifier.joblib` e classifica o sinal em tempo real a partir dos landmarks da pose.
*   **`tests/`**: Testes automatizados (`pytest`) para `utils.py` e `classifier.py`.
*   **`.github/workflows/`**: Workflow de CI que roda os testes a cada push/PR.

---

## 🚀 Como Rodar o Projeto

Todos os comandos abaixo devem ser executados a partir da **raiz do projeto** (`pose-detector/`), já que `src/` e `dataset/` são pacotes Python que se importam entre si.

1. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Garanta que você tem o modelo do MediaPipe:**
   Você precisa do arquivo `.task` dentro da pasta `models/`. Se não tiver, baixe rodando:
   ```bash
   mkdir -p models && wget -O models/pose_landmarker_lite.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
   ```

3. **Treine o classificador** (gera `models/pose_classifier.joblib`, necessário para o passo seguinte — veja detalhes na seção [Dataset e Modelo de ML](#-dataset-e-modelo-de-ml)):
   ```bash
   python -m training.train_classifier
   ```

4. **Rode o classificador em tempo real (webcam):**
   ```bash
   python -m src.main
   ```

5. **Para gravar novos dados (Criação do seu Dataset):**
   ```bash
   python -m dataset.collect_data
   ```
   *Pressione `a`, `b` ou `c` na frente da câmera para gravar 30 amostras seguidas de coordenadas da pose diretamente no arquivo CSV.*

---

## 🧪 Dataset e Modelo de ML

O `dataset.csv` atual vem de vídeos reais do [MINDS-Libras](https://www.kaggle.com/datasets/j0aopsantos/minds-libras) (Kaggle, MIT), não de gravações próprias. Para reproduzir do zero:

```bash
# 1. baixar os vídeos de um sinalizante (precisa de um token de API do Kaggle em ~/.kaggle/access_token)
kaggle datasets download j0aopsantos/minds-libras -f <arquivo>.mp4 -p dataset/raw_videos

# 2. extrair os landmarks dos vídeos para dataset/dataset.csv
python -m dataset.extract_from_videos

# 3. treinar o classificador (RandomForest) e ver as métricas
python -m training.train_classifier
```

**Resultado atual**: ~49% de acurácia média (validação cruzada de 5 folds, agrupada por vídeo — nenhum frame do mesmo vídeo aparece em treino e teste ao mesmo tempo) em 20 classes (acaso ≈ 5%). Uma primeira tentativa com split aleatório por frame (sem agrupar por vídeo) tinha dado 97% — número inflado por vazamento de dado, já que frames vizinhos do mesmo vídeo são quase idênticos. A EDA (`notebooks/eda.ipynb`) explica a causa provável da acurácia moderada: o `PoseLandmarker` só captura a pose do corpo (ombros, cotovelos, pulsos...), sem os pontos da mão — e em Libras boa parte do significado do sinal está na configuração da mão.

---

## ✅ Testes

O projeto tem testes automatizados (`pytest`) para a lógica que não depende de webcam (cálculo de ângulos e regras de classificação), rodados automaticamente via GitHub Actions a cada push/PR na `main`.

```bash
pip install -r requirements.txt
pytest -v
```

---

## 💡 Ideias Futuras

*   **Landmarks de mão**: trocar o `PoseLandmarker` pelo `HolisticLandmarker` do MediaPipe (adiciona os 21 pontos de cada mão) — deve melhorar bastante a acurácia, já que a EDA indica que a pose do corpo sozinha não separa bem sinais com trajetória de braço parecida.
*   E se, em vez de ser apenas um sistema de classificação isolado, nós pudéssemos evoluir esse modelo sequencial (treinado com LSTM) para um **sistema de tradução contínua e direta de Libras para o Português** em frases completas?

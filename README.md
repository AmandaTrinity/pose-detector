# Detector de Poses e Classificador de Libras

Este projeto utiliza **Python**, **OpenCV** e a moderna **Tasks API do MediaPipe (Google)** para detectar pontos de referência do corpo humano e da mão (landmarks) em tempo real via webcam, e classificar **sinais de palavras** (pose do corpo) e **letras do alfabeto manual** (landmarks de mão).

---

## 🏗️ Estrutura do Projeto

O projeto foi organizado de forma modular (Orientação a Objetos) para separar as responsabilidades. Palavras (pose do corpo) e letras do alfabeto (landmarks de mão) são dois pipelines paralelos, espelhados em arquivo/nome:

*   **`models/`**:
    *   `pose_landmarker_lite.task` / `hand_landmarker.task`: modelos pré-treinados do MediaPipe (versionados).
    *   `pose_classifier.joblib` / `alphabet_classifier.joblib`: nossos classificadores — **não versionados**, gerados rodando `training/train_classifier.py` / `training/train_alphabet_classifier.py`.
*   **`dataset/`**:
    *   `collect_data.py`: grava coordenadas do corpo via webcam e salva em `dataset.csv` (cada gravação vira uma "sessão" com um `video_id` próprio).
    *   `extract_from_videos.py`: roda o `PoseDetector` sobre vídeos de `raw_videos/` (não versionado — vem do Kaggle) e gera `dataset.csv` a partir deles.
    *   `schema.py`: nomes de coluna do `dataset.csv` (33 landmarks × x/y/z + `video_id` + `label`) e função `carregar_dataset()`.
    *   `dataset.csv`: dataset atual — 2970 amostras, 20 sinais de Libras, extraídas do [MINDS-Libras](https://www.kaggle.com/datasets/j0aopsantos/minds-libras) (Sinalizador01).
    *   `extract_from_alphabet_images.py`: roda o `HandDetector` (paralelizado) sobre imagens de `raw_alphabet/` (não versionado — vem do Kaggle) e gera `dataset_alfabeto.csv`, já com os landmarks normalizados (ver [Dataset e Modelo de ML](#-dataset-e-modelo-de-ml)).
    *   `alphabet_schema.py`: nomes de coluna do `dataset_alfabeto.csv` (21 landmarks de mão × x/y/z + `split` + `label`) e função `carregar_dataset()`.
    *   `dataset_alfabeto.csv`: dataset atual — 45.665 amostras, 21 letras do alfabeto manual, extraídas do [LIBRAS - Hand Landmarks Dataset](https://www.kaggle.com/datasets/williansoliveira/libras) (Kaggle).
*   **`notebooks/`**:
    *   `eda.ipynb`: análise exploratória do dataset de palavras (distribuição de classes, ranges dos landmarks, projeção PCA).
    *   `eda_alfabeto.ipynb`: análise exploratória do dataset de letras (balanceamento de classes, amostras, verificação de leakage entre treino/teste).
*   **`training/`**: `train_classifier.py` / `train_alphabet_classifier.py` — treinam um `RandomForestClassifier` e salvam o modelo + relatório de métricas.
*   **`reports/`**: saídas do treino (matrizes de confusão) — **não versionado**, regenerado a cada treino.
*   **`src/`**:
    *   `main.py` / `main_alfabeto.py`: O "maestro" de cada pipeline. Ligam a webcam, coordenam as classes e mostram o resultado na tela.
    *   `pose_detector.py` / `hand_detector.py`: Classes *Wrapper* que encapsulam e facilitam o uso da IA do MediaPipe.
    *   `utils.py`: Funções matemáticas de suporte (`angulo_corpo` — não usada mais pelo `classifier.py` atual, mas mantida com testes).
    *   `landmarks_utils.py`: `normalizar_landmarks` — normaliza os 21 landmarks de mão (origem no pulso, escala pela própria mão) pra ficarem invariantes a posição/distância da câmera. Usado tanto na extração do dataset quanto na classificação ao vivo, pra garantir que os dois usem exatamente a mesma transformação.
    *   `classifier.py` / `alphabet_classifier.py`: carregam o `.joblib` correspondente e classificam em tempo real a partir dos landmarks.
*   **`tests/`**: Testes automatizados (`pytest`) para `utils.py`, `classifier.py` e `alphabet_classifier.py`.
*   **`.github/workflows/`**: Workflow de CI que treina os dois classificadores e roda os testes a cada push/PR.

---

## 🚀 Como Rodar o Projeto

Todos os comandos abaixo devem ser executados a partir da **raiz do projeto** (`pose-detector/`), já que `src/` e `dataset/` são pacotes Python que se importam entre si.

1. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Garanta que você tem os modelos do MediaPipe:**
   `models/pose_landmarker_lite.task` e `models/hand_landmarker.task` já vêm versionados no repositório. Se por algum motivo faltar algum, baixe rodando:
   ```bash
   mkdir -p models
   wget -O models/pose_landmarker_lite.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
   wget -O models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
   ```

3. **Treine os classificadores** (gera `models/pose_classifier.joblib` e `models/alphabet_classifier.joblib`, necessários pros passos seguintes — veja detalhes na seção [Dataset e Modelo de ML](#-dataset-e-modelo-de-ml)):
   ```bash
   python -m training.train_classifier
   python -m training.train_alphabet_classifier
   ```

4. **Rode o classificador em tempo real (webcam):**
   ```bash
   python -m src.main            # reconhece palavras (pose do corpo)
   python -m src.main_alfabeto   # reconhece letras do alfabeto manual (landmarks de mão)
   ```
   O sinal/letra reconhecido aparece escrito no próprio vídeo da webcam (e também no console).

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

### Alfabeto manual

O `dataset_alfabeto.csv` vem de imagens reais do [LIBRAS - Hand Landmarks Dataset](https://www.kaggle.com/datasets/williansoliveira/libras) (Kaggle). Diferente do dataset de palavras, aqui a escolha de landmarks de **mão** (`HandDetector`/`HandLandmarker`) em vez de corpo foi deliberada desde o início, exatamente pra resolver a limitação descrita acima. Para reproduzir do zero:

```bash
# 1. baixar as imagens (pastas train/test/<letra>/*.png)
kaggle datasets download williansoliveira/libras -p dataset/raw_alphabet --unzip

# 2. extrair os landmarks de mão (normalizados) das imagens para dataset/dataset_alfabeto.csv
python -m dataset.extract_from_alphabet_images

# 3. treinar o classificador (RandomForest) e ver as métricas
python -m training.train_alphabet_classifier
```

**Resultado atual**: ~100% de acurácia no split de teste oficial do Kaggle (21 classes, acaso ≈ 5%) — bem acima dos 49% do modelo de palavras, confirmando que handshape é mesmo a feature que faltava. Duas decisões de dados que foram importantes pra chegar nesse número honestamente:

*   **Remoção de leakage entre treino/teste**: a EDA (`notebooks/eda_alfabeto.ipynb`) achou 33 imagens (0,29% do teste) duplicadas *byte-a-byte* entre os dois splits oficiais — removidas do teste antes de treinar.
*   **Normalização dos landmarks** (`src/landmarks_utils.py`): as imagens do Kaggle são recortes bem apertados (a mão ocupa o frame quase todo), então as coordenadas cruas do MediaPipe (normalizadas só pela imagem) ficam, sem querer, implicitamente normalizadas por posição/escala. Numa webcam real a mão é só uma fração do frame, numa distância variável — sem corrigir isso, o modelo classificava mal ao vivo mesmo com ótima acurácia offline (ex: confundia U com V). A normalização (origem no pulso, escala pela distância pulso → base do dedo médio) resolveu esse caso.

**Limitação conhecida em aberto**: mesmo após a normalização, o modelo ainda confunde ocasionalmente **R e U** ao vivo — esse par já era o mais confundido nos dados limpos do Kaggle (a diferença entre eles é só dedos cruzados vs. paralelos). A suspeita é sensibilidade ao eixo `z` (profundidade), que o MediaPipe estima com menos precisão que x/y, e à rotação da mão em relação à câmera — normalizamos posição e escala, mas não rotação. Próximo passo natural: normalizar também a orientação da mão (não só posição/escala) antes de extrair as features.

---

## ✅ Testes

O projeto tem testes automatizados (`pytest`) para a lógica que não depende de webcam (cálculo de ângulos e regras de classificação), rodados automaticamente via GitHub Actions a cada push/PR na `main`.

```bash
pip install -r requirements.txt
pytest -v
```

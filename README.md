# Detector de Poses e Coletor de Dados de Libras

Este projeto utiliza **Python**, **OpenCV** e a moderna **Tasks API do MediaPipe (Google)** para detectar pontos de referência do corpo humano (landmarks) em tempo real via webcam.

---

## 🏗️ Estrutura do Projeto

O projeto foi organizado de forma modular (Orientação a Objetos) para separar as responsabilidades:

*   **`models/`**: Pasta destinada ao modelo pré-treinado do MediaPipe (`pose_landmarker_lite.task`). *(Nota: Este arquivo pode não estar no GitHub devido ao tamanho).*
*   **`dataset/`**:
    *   `collect_data.py`: Script desenvolvido para gravar as coordenadas do corpo e salvar em um arquivo CSV, criando um banco de dados real para treinamento de Redes Neurais (como LSTMs).
    *   `dataset.csv`: O arquivo final com os números crus que alimentarão a IA.
*   **`src/`**:
    *   `main.py`: O "maestro" do projeto. Liga a webcam, coordena as classes e mostra o resultado na tela.
    *   `pose_detector.py`: Classe *Wrapper* que encapsula e facilita o uso da IA do MediaPipe.
    *   `utils.py`: Funções matemáticas de suporte (ex: cálculo de ângulo entre vetores 3D do corpo).
    *   `classifier.py`: Contém a lógica de negócio (ex: regras matemáticas para identificar memes específicos ou poses estáticas).
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

3. **Para rodar o classificador em tempo real (Teste de Poses):**
   ```bash
   python -m src.main
   ```

4. **Para gravar novos dados (Criação do seu Dataset):**
   ```bash
   python -m dataset.collect_data
   ```
   *Pressione `a`, `b` ou `c` na frente da câmera para gravar 30 amostras seguidas de coordenadas da pose diretamente no arquivo CSV.*

---

## ✅ Testes

O projeto tem testes automatizados (`pytest`) para a lógica que não depende de webcam (cálculo de ângulos e regras de classificação), rodados automaticamente via GitHub Actions a cada push/PR na `main`.

```bash
pip install -r requirements.txt
pytest -v
```

---

## 💡 Ideias Futuras
E se, em vez de ser apenas um sistema de classificação isolado, nós pudéssemos evoluir esse modelo sequencial (treinado com LSTM) para um **sistema de tradução contínua e direta de Libras para o Português** em frases completas?

# Pipeline de Treinamento com MLflow - Parkinson's Disease

Este repositório contém a implementação da atividade "Pipeline de treino e avaliação com MLflow". O objetivo do projeto é prever a severidade motora da Doença de Parkinson (`motor_UPDRS`) com base em atributos extraídos de gravações de voz.

## 🎯 Objetivo e Perguntas dos Experimentos

Foram realizados vários experimentos com Redes Neurais (*Multilayer Perceptron*) em PyTorch para entender a dinâmica de aprendizado sobre o dataset. 

As principais perguntas objetivas que guiaram as configurações foram:
1. **Experimento 1 - Base (Exp1_PyTorch_Base):** *"Qual o erro residual nativo e o nível de overfitting de uma rede neural simples sem técnicas de regularização ao tentar prever a gravidade da doença motora?"*
2. **Experimento 2 - Regularizado (Exp2_PyTorch_Regularized):** *"O uso agressivo de Dropout (60%) e decaimento de pesos severo (L2 penalty) restringe demais a rede causando underfitting num dataset clínico?"*
3. **Experimento 3 - Ótimo (Exp3_PyTorch_Optimized):** *"Ao equilibrar a regularização (redução do Dropout para 10% e relaxamento do L2 penalty), é possível anular a variância de generalização e otimizar a previsão final sem estagnar a rede?"*

## 🧩 Arquitetura do Modelo e Configurações
Todos os experimentos foram padronizados para utilizar a mesma arquitetura de neurônios nas camadas ocultas (`256 -> 128 -> 64 -> 1`). 
As variáveis ajustadas em cada experimento (cadastradas no MLflow via `mlflow.log_params`) foram:
- **Taxa de aprendizado (Learning Rate)**
- **Épocas (Epochs)**
- **Técnicas de regularização (Dropout e Weight Decay)**

## 📊 Dataset Utilizado e Divisão
* **Parkinson's Telemonitoring Dataset**: Contém métricas de voz (Jitter, Shimmer, NHR, HNR, DFA, PPE) extraídas de gravações de pacientes com Parkinson. 
* **Variável Alvo:** `motor_UPDRS` (Regressão contínua).
* **Pré-processamento:** Normalização via `StandardScaler`.
* **Estratégia de Divisão:** Os dados foram separados em três conjuntos para garantir uma avaliação sem viés:
  - **Treino:** ~72%
  - **Validação:** ~15%
  - **Teste:** ~15%

## 📈 Rastreio e Evidências no MLflow
O projeto utiliza o MLflow para rastrear ponta a ponta o desenvolvimento do modelo (Configuração, Dados, Execução e Evidências):
- **Execução:** Semente global (`SEED=42`) para reprodutibilidade. O ambiente (`conda.yaml` e dependências) é capturado automaticamente pelo MLflow.
- **Métricas Passo-a-Passo:** As métricas de erro (`train_rmse` e `val_rmse`) são rastreadas e logadas a cada 20 épocas, gerando curvas visuais dinâmicas direto na interface do MLflow.
- **Avaliação Final:** O modelo de cada experimento é submetido ao conjunto isolado de Teste, gerando a métrica final `test_rmse_final`.
- **Artefatos:** O gráfico comparativo estático (Curvas de Treino e Validação) é exportado e anexado aos artefatos da *run* automaticamente (`mlflow.log_artifact`). O modelo PyTorch serializado (`.pt2`) também é salvo.

## ⚙️ Estrutura do Pipeline

A atividade foi organizada utilizando um pipeline estruturado de ponta a ponta:

- `train_pytorch.py`: Script avançado desenvolvido para configurar, treinar, e rastrear a MLP.
- `mlruns/` & `mlflow.db`: Rastros de execução, métricas e exportação de gráficos de loss por época.

## 🚀 Como Executar

Siga o passo a passo abaixo para reproduzir os experimentos na sua máquina:

**1. Clone o repositório e acesse a pasta:**
```bash
git clone [URL_DO_SEU_REPOSITORIO]
cd projeto-mlflow-parkinsons
```

**2. Crie e ative um ambiente virtual (Recomendado):**
```bash
python -m venv .venv
# No Windows:
.venv\Scripts\activate
# No Linux/Mac:
source .venv/bin/activate
```

**3. Instale as dependências:**
Certifique-se de que o arquivo `requirements.txt` está na raiz do projeto e execute:
```bash
pip install -r requirements.txt
```

**4. Dataset:**
O arquivo de dados `parkinsons_updrs.data` (separado por vírgulas) deve estar na mesma pasta do script principal. O nosso script já faz a leitura e a quebra (Treino/Validação/Teste) automaticamente.

**5. Inicie o servidor do MLflow:**
Para conseguir visualizar os painéis e gráficos, inicie o servidor do MLflow em um terminal separado:
```bash
mlflow ui --host 0.0.0.0 --port 5000
```
*Após rodar o comando, abra o seu navegador e acesse: [http://localhost:5000](http://localhost:5000)*

**6. Execute os Testes (Treinamento do Pipeline):**
Em outro terminal (com o ambiente virtual ativado), execute o pipeline principal. O script orquestra os 3 experimentos, testa eles separadamente e envia tudo para o MLflow:
```bash
python train_pytorch.py
```

## 🎥 Evidências e Entregáveis

- **Vídeo de Apresentação:** [Insira o link do vídeo aqui]
- **Repositório GitHub:** [Insira o link do repositório aqui]

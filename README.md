# 🤖 ML Studio – Análise e Modelagem de Dados com Streamlit

Este projeto oferece uma aplicação web interativa usando **Streamlit** para realizar todo o pipeline de *machine learning*, desde o upload e análise dos dados até o treinamento, avaliação e previsão com modelos.

---

## ✅ Funcionalidades

- Upload de datasets CSV ou uso de dataset de exemplo (Spotify)
- Análise exploratória com gráficos, estatísticas e correlações
- Seleção de variáveis e tipo de tarefa (Classificação, Regressão ou Clustering)
- Treinamento automatizado com **PyCaret**
- Avaliação com métricas (accuracy, RMSE, R², matriz de confusão)
- Gráficos interativos com Plotly
- Previsões com entrada manual ou em lote via CSV

---

## 📁 Dataset de Exemplo

Este projeto inclui um dataset real de músicas do Spotify com diversas features como energia, dançabilidade, popularidade, entre outras.

📂 Caminho: `data/SpotifyFeatures.csv`

Se você não quiser subir seu próprio CSV, basta selecionar a opção **"Usar dataset do Spotify (exemplo)"** dentro da aplicação.
"https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db"

---

## ⚙️ Requisitos

- **Python 3.8 ou superior**

Instale as dependências com:

```bash
pip install -r requirements.txt
```

Conteúdo do `requirements.txt`:

```
streamlit
pandas
pycaret
matplotlib
seaborn
ydata-profiling
dtale
kaggle
```

---

## 🚀 Como rodar o projeto

1. Clone o repositório:

```bash
git clone https://github.com/Santanadev5/An-lise_Modelagem_machine_learning.git
cd An-lise_Modelagem_machine_learning
```

2. Instale os requisitos:

```bash
pip install -r requirements.txt
```

3. Execute a aplicação Streamlit:

```bash
python -m streamlit run streamlit_app.py
```

Abra no navegador: [http://localhost:8501](http://localhost:8501)

---

## 🧪 Como usar a aplicação

1. Use o menu lateral da aplicação para navegar entre:
   - 📊 **Upload e Análise de Dados**
   - 🔍 **Análise Exploratória**
   - ⚙️ **Configuração do Modelo**
   - 🎯 **Treinamento e Avaliação**
   - 🔮 **Previsões**

2. Carregue seu dataset ou use o dataset do Spotify
3. Escolha as variáveis e o tipo de tarefa
4. Treine e avalie seu modelo
5. Faça previsões com novos dados

---

## 📂 Estrutura do Projeto

```
An-lise_Modelagem_machine_learning/
├── streamlit_app.py           ← Aplicação principal em Streamlit
├── requirements.txt           ← Dependências do projeto
├── data/                      ← Datasets de exemplo
│   └── SpotifyFeatures.csv
└── README.md                  ← Este arquivo
```

---

## 🧠 Modelos utilizados com PyCaret

- Classificação: `RandomForest`, `XGBoost`, `LogisticRegression`, etc.
- Regressão: `LinearRegression`, `GradientBoosting`, `XGBoost`, etc.
- Clustering: `KMeans` (com visualização 2D automática)

---

## 📦 Exportação de Resultados

- Baixe arquivos de previsão em CSV
- Visualize métricas e gráficos diretamente na interface
- Suporte a entrada manual e em lote

---

## 📄 Licença

Este projeto é open source sob a licença MIT.

---


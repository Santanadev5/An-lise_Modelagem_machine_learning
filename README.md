# PyCaret Spotify ML Pipeline 🎧

Este projeto realiza análise exploratória de dados (EDA), seleção/alteração de variáveis e teste de modelos de machine learning usando PyCaret, com base no dataset `SpotifyFeatures.csv`.

## ✅ Funcionalidades atendidas

✔️ Análise exploratória de dados (EDA) com `ydata-profiling`  
✔️ Seleção e tratamento de variáveis com `PyCaret`  
✔️ Comparação automática de modelos de machine learning  
✔️ Geração de relatório HTML interativo com gráficos e estatísticas

---

## ⚙️ Requisitos

- Python **3.10** ou superior
- Bibliotecas:
  ```bash
  pip install -r requirements.txt
  ```
  > Dica: certifique-se de instalar `pycaret`, `pandas`, `matplotlib`, `seaborn` e `ydata-profiling`

---

## 🚀 Como rodar o projeto

1. Clone o repositório e entre na pasta do projeto:
   ```bash
   git clone https://github.com/Santanadev5/Dataset-Spotify.git
   cd Dataset-Spotify
   ```

2. Para **gerar a análise exploratória em HTML**:
   ```bash
   python main.py profile SpotifyFeatures.csv
   ```
   Isso irá gerar um arquivo chamado `profile_report.html` na raiz do projeto.

3. Para **treinar o modelo com PyCaret (classificação)**:
   ```bash
   python main.py train SpotifyFeatures.csv genre classification
   ```
   O melhor modelo será selecionado automaticamente com `compare_models()`.

---

## 🌐 Como abrir o relatório HTML

Após gerar o `profile_report.html`, você pode abrir de 2 formas:

### ➤ Manualmente:
1. Vá até a pasta do projeto pelo Explorador de Arquivos
2. Clique duas vezes no arquivo `profile_report.html`

### ➤ Via terminal PowerShell:
```bash
start msedge "$PWD\profile_report.html"
```
> Ou substitua `msedge` por `chrome` se quiser abrir com o Google Chrome

### ➤ Via Python:
```python
import webbrowser
webbrowser.open("profile_report.html")
```
Execute isso após entrar no modo `python` no terminal.

---

## 📁 Estrutura do Projeto

```
Dataset-Spotify/
├── application/
│   ├── data/SpotifyFeatures.csv   ← Base de dados
│   └── use_cases.py               ← Lógica dos comandos
├── adapters/
│   └── pycaret_adapter.py         ← Treinamento com PyCaret
├── ports/
│   └── training_port.py           ← Interface de treino
├── profile_report.html            ← Relatório EDA gerado
├── correlation_matrix.png         ← Gráfico de correlação
├── main.py                        ← CLI principal
└── README.md                      ← Este arquivo
```

---

## ✍️ Autor

Nicolas Santana  
Orientação: Profª Thayse  
Trabalho acadêmico - Database Spotify Tracks DB -> https://www.kaggle.com/datasets/zaheenhamidani/ultimate-spotify-tracks-db <-

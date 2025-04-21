import pandas as pd
from ports.training_port import TrainingPort

# PyCaret tasks
from pycaret.classification import setup as class_setup, compare_models as class_compare, get_config as class_config
from pycaret.regression import setup as reg_setup, compare_models as reg_compare, get_config as reg_config
from pycaret.clustering import setup as clus_setup, create_model as clus_create, get_config as clus_config

import seaborn as sns
import matplotlib.pyplot as plt

class PyCaretAdapter(TrainingPort):
    def train_model(self, df: pd.DataFrame, target: str, task_type: str):
        """
        Faz análise exploratória, seleção de variáveis e treinamento de modelos com PyCaret.
        """
        print("\n📊 Iniciando Análise Exploratória de Dados (EDA)...")
        print("Resumo estatístico:\n", df.describe())
        print("Valores nulos por coluna:\n", df.isnull().sum())

        # Correlação entre variáveis numéricas
        if task_type in ["classification", "regression"]:
            plt.figure(figsize=(10, 8))
            corr = df.corr(numeric_only=True)
            sns.heatmap(corr, annot=True, cmap="coolwarm")
            plt.title("Matriz de Correlação")
            plt.tight_layout()
            plt.savefig("correlation_matrix.png")
            print("✅ Gráfico de correlação salvo como 'correlation_matrix.png'.")

        print("\n⚙️ Iniciando Setup com PyCaret...")

        if task_type == "classification":
            class_setup(data=df, target=target, session_id=123, html=False, silent=True, verbose=False)
            print("🔍 Variáveis selecionadas:", class_config("X").columns.tolist())
            best_model = class_compare()
            print("✅ Melhor modelo de Classificação:", best_model)
            return best_model

        elif task_type == "regression":
            reg_setup(data=df, target=target, session_id=123, html=False, silent=True, verbose=False)
            print("🔍 Variáveis selecionadas:", reg_config("X").columns.tolist())
            best_model = reg_compare()
            print("✅ Melhor modelo de Regressão:", best_model)
            return best_model

        elif task_type == "clustering":
            clus_setup(data=df, session_id=123, html=False, silent=True, verbose=False)
            print("🔍 Variáveis utilizadas:", clus_config("X").columns.tolist())
            best_model = clus_create("kmeans")
            print("✅ Modelo de Clustering criado:", best_model)
            return best_model

        else:
            raise ValueError("❌ task_type inválido. Escolha entre: classification, regression ou clustering.")
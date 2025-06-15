import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import warnings
warnings.filterwarnings("ignore")

# Configuração da página
st.set_page_config(
    page_title="ML Studio - Análise e Modelagem",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e8b57;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def show_home_page():
    """Página inicial com informações sobre a aplicação"""
    st.markdown("<h2 class=\"section-header\">Bem-vindo ao ML Studio!</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    Esta aplicação oferece uma interface completa para análise de dados e machine learning, incluindo:
    
    ### 🎯 Funcionalidades Principais:
    - **Upload de Dados**: Carregue seus datasets em formato CSV
    - **Análise Exploratória**: Visualize estatísticas, correlações e distribuições
    - **Seleção de Variáveis**: Escolha as features mais relevantes para seu modelo
    - **Modelagem ML**: Treine modelos de classificação, regressão ou clustering
    - **Avaliação**: Analise métricas de performance dos modelos
    - **Previsões**: Faça previsões com novos dados
    
    ### 🛠️ Tecnologias Utilizadas:
    - **Streamlit**: Interface web interativa
    - **PyCaret**: Framework de machine learning automatizado
    - **Pandas**: Manipulação de dados
    - **Plotly/Matplotlib**: Visualizações interativas
    
    ### 🚀 Como Começar:
    1. Vá para "📊 Upload e Análise de Dados" para carregar seu dataset
    2. Explore seus dados na seção "🔍 Análise Exploratória"
    3. Configure seu modelo em "⚙️ Configuração do Modelo"
    4. Treine e avalie em "🎯 Treinamento e Avaliação"
    5. Faça previsões em "🔮 Previsões"
    """)
    
    # Informações sobre o dataset de exemplo
    st.markdown("<h3 class=\"section-header\">📁 Dataset de Exemplo</h3>", unsafe_allow_html=True)
    st.info("""
    💡 **Dica**: Se você não tem um dataset próprio, pode usar o dataset do Spotify que já está incluído no projeto!
    Vá para a seção de upload e carregue o arquivo de exemplo.
    """)

def show_data_upload_page():
    """Página para upload e visualização inicial dos dados"""
    st.markdown("<h2 class=\"section-header\">📊 Upload e Análise de Dados</h2>", unsafe_allow_html=True)
    
    # Opções de carregamento
    upload_option = st.radio(
        "Como você gostaria de carregar os dados?",
        ["📁 Upload de arquivo CSV", "🎵 Usar dataset do Spotify (exemplo)"]
    )
    
    if upload_option == "📁 Upload de arquivo CSV":
        uploaded_file = st.file_uploader(
            "Escolha um arquivo CSV",
            type=["csv"],
            help="Carregue um arquivo CSV com seus dados para análise"
        )
        
        if uploaded_file is not None:
            try:
                # Ler o arquivo
                df = pd.read_csv(uploaded_file)
                st.session_state.df = df
                st.success(f"✅ Arquivo carregado com sucesso! {df.shape[0]} linhas e {df.shape[1]} colunas.")
                
            except Exception as e:
                st.error(f"❌ Erro ao carregar o arquivo: {str(e)}")
                return
    
    elif upload_option == "🎵 Usar dataset do Spotify (exemplo)":
        try:
            # Tentar carregar o dataset do Spotify
            spotify_files = [
                "data/SpotifyFeatures.csv",
                "data/spotify_songs.csv",
                "data/tracks.csv",
                "data/dataset.csv"
            ]
            
            loaded = False
            for file_path in spotify_files:
                try:
                    df = pd.read_csv(file_path)
                    st.session_state.df = df
                    st.success(f"✅ Dataset do Spotify carregado! {df.shape[0]} linhas e {df.shape[1]} colunas.")
                    loaded = True
                    break
                except:
                    continue
            
            if not loaded:
                st.warning("⚠️ Dataset do Spotify não encontrado. Por favor, faça upload de um arquivo CSV.")
                return
                
        except Exception as e:
            st.error(f"❌ Erro ao carregar dataset do Spotify: {str(e)}")
            return
    
    # Mostrar informações do dataset se carregado
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Informações básicas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📊 Linhas</h4>
                <h2>{df.shape[0]:,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📋 Colunas</h4>
                <h2>{df.shape[1]:,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h4>🔢 Numéricas</h4>
                <h2>{len(df.select_dtypes(include=[np.number]).columns)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h4>📝 Categóricas</h4>
                <h2>{len(df.select_dtypes(include=["object"]).columns)}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Preview dos dados
        st.markdown("<h3 class=\"section-header\">👀 Preview dos Dados</h3>", unsafe_allow_html=True)
        st.dataframe(df.head(10), use_container_width=True)
        
        # Informações sobre tipos de dados
        st.markdown("<h3 class=\"section-header\">📋 Informações das Colunas</h3>", unsafe_allow_html=True)
        
        col_info = pd.DataFrame({
            "Coluna": df.columns,
            "Tipo": df.dtypes,
            "Valores Únicos": [df[col].nunique() for col in df.columns],
            "Valores Nulos": [df[col].isnull().sum() for col in df.columns],
            "% Nulos": [round(df[col].isnull().sum() / len(df) * 100, 2) for col in df.columns]
        })
        
        st.dataframe(col_info, use_container_width=True)
        
        # Botão para próxima etapa
        if st.button("➡️ Prosseguir para Análise Exploratória", type="primary"):
            st.success("✅ Dados carregados! Vá para a seção \'Análise Exploratória\' no menu lateral.")

def show_exploratory_analysis_page():
    """Página para análise exploratória de dados"""
    st.markdown("<h2 class=\"section-header\">🔍 Análise Exploratória de Dados</h2>", unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("⚠️ Nenhum dataset carregado. Por favor, vá para a seção \'Upload e Análise de Dados\' primeiro.")
        return
    
    df = st.session_state.df
    
    # Tabs para diferentes tipos de análise
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Estatísticas Descritivas", 
        "📈 Distribuições", 
        "🔗 Correlações", 
        "🔍 Valores Ausentes",
        "📋 Análise de Variáveis"
    ])
    
    with tab1:
        st.markdown("<h3 class=\"section-header\">📊 Estatísticas Descritivas</h3>", unsafe_allow_html=True)
        
        # Estatísticas para variáveis numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.markdown("**Variáveis Numéricas:**")
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        
        # Estatísticas para variáveis categóricas
        categorical_cols = df.select_dtypes(include=["object"]).columns
        if len(categorical_cols) > 0:
            st.markdown("**Variáveis Categóricas:**")
            cat_stats = pd.DataFrame({
                "Coluna": categorical_cols,
                "Valores Únicos": [df[col].nunique() for col in categorical_cols],
                "Valor Mais Frequente": [df[col].mode().iloc[0] if len(df[col].mode()) > 0 else "N/A" for col in categorical_cols],
                "Frequência do Mais Comum": [df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0 for col in categorical_cols]
            })
            st.dataframe(cat_stats, use_container_width=True)
    
    with tab2:
        st.markdown("<h3 class=\"section-header\">📈 Distribuições das Variáveis</h3>", unsafe_allow_html=True)
        
        # Seleção de variável para análise
        if len(numeric_cols) > 0:
            selected_numeric = st.selectbox("Selecione uma variável numérica:", numeric_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Histograma
                fig_hist = px.histogram(
                    df, 
                    x=selected_numeric, 
                    title=f"Distribuição de {selected_numeric}",
                    nbins=30
                )
                fig_hist.update_layout(height=400)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Box plot
                fig_box = px.box(
                    df, 
                    y=selected_numeric, 
                    title=f"Box Plot de {selected_numeric}"
                )
                fig_box.update_layout(height=400)
                st.plotly_chart(fig_box, use_container_width=True)
        
        # Análise de variáveis categóricas
        if len(categorical_cols) > 0:
            selected_categorical = st.selectbox("Selecione uma variável categórica:", categorical_cols)
            
            # Gráfico de barras para variáveis categóricas
            value_counts = df[selected_categorical].value_counts().head(20)
            fig_bar = px.bar(
                x=value_counts.index, 
                y=value_counts.values,
                title=f"Distribuição de {selected_categorical} (Top 20)",
                labels={"x": selected_categorical, "y": "Frequência"}
            )
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab3:
        st.markdown("<h3 class=\"section-header\">🔗 Análise de Correlações</h3>", unsafe_allow_html=True)
        
        if len(numeric_cols) > 1:
            # Matriz de correlação
            corr_matrix = df[numeric_cols].corr()
            
            # Heatmap de correlação
            fig_corr = px.imshow(
                corr_matrix,
                title="Matriz de Correlação",
                color_continuous_scale="RdBu",
                aspect="auto"
            )
            fig_corr.update_layout(height=600)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Correlações mais altas
            st.markdown("**Correlações mais fortes (> 0.5 ou < -0.5):**")
            
            # Encontrar correlações altas
            high_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5:
                        high_corr.append({
                            "Variável 1": corr_matrix.columns[i],
                            "Variável 2": corr_matrix.columns[j],
                            "Correlação": round(corr_val, 3)
                        })
            
            if high_corr:
                high_corr_df = pd.DataFrame(high_corr)
                high_corr_df = high_corr_df.sort_values("Correlação", key=abs, ascending=False)
                st.dataframe(high_corr_df, use_container_width=True)
            else:
                st.info("Nenhuma correlação forte encontrada (> 0.5 ou < -0.5)")
        else:
            st.info("Necessário pelo menos 2 variáveis numéricas para análise de correlação.")
    
    with tab4:
        st.markdown("<h3 class=\"section-header\">🔍 Análise de Valores Ausentes</h3>", unsafe_allow_html=True)
        
        # Contagem de valores ausentes
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        
        missing_df = pd.DataFrame({
            "Coluna": missing_data.index,
            "Valores Ausentes": missing_data.values,
            "Percentual (%)": missing_percent.values
        })
        missing_df = missing_df[missing_df["Valores Ausentes"] > 0].sort_values("Valores Ausentes", ascending=False)
        
        if len(missing_df) > 0:
            st.dataframe(missing_df, use_container_width=True)
            
            # Gráfico de valores ausentes
            fig_missing = px.bar(
                missing_df, 
                x="Coluna", 
                y="Percentual (%)",
                title="Percentual de Valores Ausentes por Coluna"
            )
            fig_missing.update_layout(height=400)
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.success("✅ Nenhum valor ausente encontrado no dataset!")
    
    with tab5:
        st.markdown("<h3 class=\"section-header\">📋 Análise Detalhada de Variáveis</h3>", unsafe_allow_html=True)
        
        # Seleção de variável para análise detalhada
        all_columns = df.columns.tolist()
        selected_column = st.selectbox("Selecione uma variável para análise detalhada:", all_columns)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Informações sobre \'{selected_column}\':**")
            
            info_data = {
                "Tipo de Dados": str(df[selected_column].dtype),
                "Valores Únicos": df[selected_column].nunique(),
                "Valores Nulos": df[selected_column].isnull().sum(),
                "Percentual Nulos": f"{(df[selected_column].isnull().sum() / len(df)) * 100:.2f}%"
            }
            
            if df[selected_column].dtype in ["int64", "float64"]:
                info_data.update({
                    "Mínimo": df[selected_column].min(),
                    "Máximo": df[selected_column].max(),
                    "Média": round(df[selected_column].mean(), 3),
                    "Mediana": df[selected_column].median(),
                    "Desvio Padrão": round(df[selected_column].std(), 3)
                })
            
            for key, value in info_data.items():
                st.metric(key, value)
        
        with col2:
            st.markdown(f"**Valores mais frequentes em \'{selected_column}\':**")
            
            value_counts = df[selected_column].value_counts().head(10)
            value_counts_df = pd.DataFrame({
                "Valor": value_counts.index,
                "Frequência": value_counts.values,
                "Percentual (%)": round((value_counts.values / len(df)) * 100, 2)
            })
            
            st.dataframe(value_counts_df, use_container_width=True)

def show_model_configuration_page():
    """Página para configuração do modelo de ML"""
    st.markdown("<h2 class=\"section-header\">⚙️ Configuração do Modelo</h2>", unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("⚠️ Nenhum dataset carregado. Por favor, vá para a seção \'Upload e Análise de Dados\' primeiro.")
        return
    
    df = st.session_state.df
    
    # Seleção do tipo de tarefa
    st.markdown("<h3 class=\"section-header\">🎯 Tipo de Tarefa de Machine Learning</h3>", unsafe_allow_html=True)
    
    task_type = st.selectbox(
        "Selecione o tipo de tarefa:",
        ["Classificação", "Regressão", "Clustering"],
        help="Escolha o tipo de problema que você quer resolver"
    )
    
    # Mapear para valores do PyCaret
    task_mapping = {
        "Classificação": "classification",
        "Regressão": "regression", 
        "Clustering": "clustering"
    }
    
    st.session_state.task_type = task_mapping[task_type]
    
    # Configuração específica por tipo de tarefa
    if task_type in ["Classificação", "Regressão"]:
        st.markdown("<h3 class=\"section-header\">🎯 Seleção da Variável Alvo</h3>", unsafe_allow_html=True)
        
        # Filtrar colunas apropriadas para cada tipo
        if task_type == "Classificação":
            # Para classificação, mostrar colunas categóricas e numéricas com poucos valores únicos
            suitable_cols = []
            for col in df.columns:
                if df[col].dtype == "object" or df[col].nunique() <= 20:
                    suitable_cols.append(col)
        else:  # Regressão
            # Para regressão, mostrar apenas colunas numéricas
            suitable_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(suitable_cols) > 0:
            target_column = st.selectbox(
                "Selecione a variável alvo (target):",
                suitable_cols,
                help=f"Para {task_type.lower()}, selecione a variável que você quer prever"
            )
            
            st.session_state.target_column = target_column
            
            # Mostrar informações sobre a variável alvo
            st.markdown(f"**Informações sobre \'{target_column}\':**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Valores Únicos", df[target_column].nunique())
            
            with col2:
                st.metric("Valores Nulos", df[target_column].isnull().sum())
            
            with col3:
                if df[target_column].dtype in ["int64", "float64"]:
                    st.metric("Média", round(df[target_column].mean(), 3))
                else:
                    most_common = df[target_column].mode().iloc[0] if len(df[target_column].mode()) > 0 else "N/A"
                    st.metric("Mais Frequente", most_common)
            
            # Distribuição da variável alvo
            if task_type == "Classificação":
                st.markdown("**Distribuição da Variável Alvo:**")
                target_counts = df[target_column].value_counts()
                fig_target = px.bar(
                    x=target_counts.index,
                    y=target_counts.values,
                    title=f"Distribuição de {target_column}",
                    labels={"x": target_column, "y": "Frequência"}
                )
                st.plotly_chart(fig_target, use_container_width=True)
            else:  # Regressão
                st.markdown("**Distribuição da Variável Alvo:**")
                fig_target = px.histogram(
                    df,
                    x=target_column,
                    title=f"Distribuição de {target_column}",
                    nbins=30
                )
                st.plotly_chart(fig_target, use_container_width=True)
        else:
            st.error(f"❌ Nenhuma coluna adequada encontrada para {task_type.lower()}.")
            return
    
    else:  # Clustering
        st.info("💡 Para clustering, todas as variáveis numéricas serão utilizadas automaticamente.")
        st.session_state.target_column = None
    
    # Seleção de features
    st.markdown("<h3 class=\"section-header\">🔧 Seleção de Variáveis (Features)</h3>", unsafe_allow_html=True)
    
    # Obter todas as colunas exceto a target (se houver)
    available_features = df.columns.tolist()
    if st.session_state.target_column:
        available_features = [col for col in available_features if col != st.session_state.target_column]
    
    # Separar por tipo
    numeric_features = df[available_features].select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df[available_features].select_dtypes(include=["object"]).columns.tolist()
    
    # Interface para seleção de features
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Variáveis Numéricas:**")
        selected_numeric = st.multiselect(
            "Selecione as variáveis numéricas:",
            numeric_features,
            default=numeric_features[:10] if len(numeric_features) > 10 else numeric_features,
            help="Selecione as variáveis numéricas que serão usadas no modelo"
        )
    
    with col2:
        st.markdown("**Variáveis Categóricas:**")
        selected_categorical = st.multiselect(
            "Selecione as variáveis categóricas:",
            categorical_features,
            default=categorical_features[:5] if len(categorical_features) > 5 else categorical_features,
            help="Selecione as variáveis categóricas que serão usadas no modelo"
        )
    
    # Combinar features selecionadas
    selected_features = selected_numeric + selected_categorical
    st.session_state.selected_features = selected_features
    
    # Mostrar resumo da seleção
    if selected_features:
        st.markdown("<h3 class=\"section-header\">📋 Resumo da Configuração</h3>", unsafe_allow_html=True)
        
        config_summary = {
            "Tipo de Tarefa": task_type,
            "Variável Alvo": st.session_state.target_column if st.session_state.target_column else "N/A (Clustering)",
            "Total de Features": len(selected_features),
            "Features Numéricas": len(selected_numeric),
            "Features Categóricas": len(categorical_features)
        }
        
        for key, value in config_summary.items():
            st.metric(key, value)
        
        # Mostrar features selecionadas
        st.markdown("**Features Selecionadas:**")
        features_df = pd.DataFrame({
            "Feature": selected_features,
            "Tipo": [df[col].dtype for col in selected_features],
            "Valores Únicos": [df[col].nunique() for col in selected_features],
            "Valores Nulos": [df[col].isnull().sum() for col in selected_features]
        })
        st.dataframe(features_df, use_container_width=True)
        
        # Botão para prosseguir
        if st.button("➡️ Prosseguir para Treinamento", type="primary"):
            st.success("✅ Configuração salva! Vá para a seção \'Treinamento e Avaliação\' no menu lateral.")
    
    else:
        st.warning("⚠️ Selecione pelo menos uma feature para continuar.")

def show_training_page():
    """Página para treinamento e avaliação do modelo"""
    st.markdown("<h2 class=\"section-header\">🎯 Treinamento e Avaliação do Modelo</h2>", unsafe_allow_html=True)
    
    if st.session_state.df is None:
        st.warning("⚠️ Nenhum dataset carregado. Por favor, configure o modelo primeiro.")
        return
    
    if st.session_state.task_type is None or st.session_state.selected_features is None:
        st.warning("⚠️ Configure o modelo primeiro na seção \'Configuração do Modelo\'.")
        return
    
    df = st.session_state.df
    
    # Preparar dados para treinamento
    if st.session_state.target_column:
        # Para classificação e regressão
        features_data = df[st.session_state.selected_features + [st.session_state.target_column]].copy()
    else:
        # Para clustering
        features_data = df[st.session_state.selected_features].copy()
    
    # Remover linhas com valores nulos
    initial_rows = len(features_data)
    features_data = features_data.dropna()
    final_rows = len(features_data)
    
    if initial_rows != final_rows:
        st.info(f"ℹ️ Removidas {initial_rows - final_rows} linhas com valores nulos. Dataset final: {final_rows} linhas.")
    
    # Mostrar configuração atual
    st.markdown("<h3 class=\"section-header\">📋 Configuração Atual</h3>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Tipo de Tarefa", st.session_state.task_type.title())
    
    with col2:
        st.metric("Linhas para Treino", final_rows)
    
    with col3:
        st.metric("Features", len(st.session_state.selected_features))
    
    with col4:
        target_display = st.session_state.target_column if st.session_state.target_column else "N/A"
        st.metric("Variável Alvo", target_display)
    
    # Botão para iniciar treinamento
    if st.button("🚀 Iniciar Treinamento", type="primary"):
        
        with st.spinner("🔄 Treinando modelos... Isso pode levar alguns minutos."):
            try:
                # Importar PyCaret baseado no tipo de tarefa
                if st.session_state.task_type == "classification":
                    from pycaret.classification import setup, compare_models, finalize_model, evaluate_model, get_config
                elif st.session_state.task_type == "regression":
                    from pycaret.regression import setup, compare_models, finalize_model, evaluate_model, get_config
                else:  # clustering
                    from pycaret.clustering import setup, create_model, assign_model, get_config
                
                # Setup do PyCaret
                if st.session_state.task_type in ["classification", "regression"]:
                    ml_setup = setup(
                        data=features_data,
                        target=st.session_state.target_column,
                        session_id=123,
                        train_size=0.8,
                        silent=True,
                        html=False,
                        verbose=False
                    )
                    
                    # Comparar modelos
                    best_models = compare_models(
                        include=["lr", "rf", "et", "gbr" if st.session_state.task_type == "regression" else "gbc", "xgboost"],
                        sort="RMSE" if st.session_state.task_type == "regression" else "Accuracy",
                        n_select=3,
                        verbose=False
                    )
                    
                    # Finalizar o melhor modelo
                    if isinstance(best_models, list):
                        best_model = finalize_model(best_models[0])
                    else:
                        best_model = finalize_model(best_models)
                    
                    st.session_state.model = best_model
                    
                else:  # clustering
                    ml_setup = setup(
                        data=features_data,
                        session_id=123,
                        silent=True,
                        html=False,
                        verbose=False
                    )
                    
                    # Criar modelo de clustering
                    kmeans_model = create_model("kmeans", num_clusters=3)
                    
                    # Atribuir clusters
                    clustered_data = assign_model(kmeans_model)
                    
                    st.session_state.model = kmeans_model
                    st.session_state.clustered_data = clustered_data
                
                st.success("✅ Treinamento concluído com sucesso!")
                
                # Mostrar resultados
                st.markdown("<h3 class=\"section-header\">📊 Resultados do Treinamento</h3>", unsafe_allow_html=True)
                
                if st.session_state.task_type in ["classification", "regression"]:
                    # Métricas do modelo
                    st.markdown("**Melhor Modelo Treinado:**")
                    st.write(f"Algoritmo: {type(best_model).__name__}")
                    
                    # Avaliação do modelo
                    try:
                        # Obter métricas do setup
                        X_test = get_config("X_test")
                        y_test = get_config("y_test")
                        
                        if st.session_state.task_type == "classification":
                            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
                            
                            y_pred = best_model.predict(X_test)
                            accuracy = accuracy_score(y_test, y_pred)
                            
                            st.metric("Acurácia no Teste", f"{accuracy:.3f}")
                            
                            # Matriz de confusão
                            cm = confusion_matrix(y_test, y_pred)
                            fig_cm = px.imshow(
                                cm,
                                title="Matriz de Confusão",
                                labels=dict(x="Predito", y="Real"),
                                color_continuous_scale="Blues"
                            )
                            st.plotly_chart(fig_cm, use_container_width=True)
                            
                        else:  # regression
                            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
                            
                            y_pred = best_model.predict(X_test)
                            mse = mean_squared_error(y_test, y_pred)
                            rmse = np.sqrt(mse)
                            r2 = r2_score(y_test, y_pred)
                            mae = mean_absolute_error(y_test, y_pred)
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("R² Score", f"{r2:.3f}")
                            with col2:
                                st.metric("RMSE", f"{rmse:.3f}")
                            with col3:
                                st.metric("MAE", f"{mae:.3f}")
                            with col4:
                                st.metric("MSE", f"{mse:.3f}")
                            
                            # Gráfico de predições vs real
                            fig_pred = px.scatter(
                                x=y_test,
                                y=y_pred,
                                title="Predições vs Valores Reais",
                                labels={"x": "Valores Reais", "y": "Predições"}
                            )
                            fig_pred.add_shape(
                                type="line",
                                x0=y_test.min(),
                                y0=y_test.min(),
                                x1=y_test.max(),
                                y1=y_test.max(),
                                line=dict(color="red", dash="dash")
                            )
                            st.plotly_chart(fig_pred, use_container_width=True)
                    
                    except Exception as e:
                        st.warning(f"⚠️ Não foi possível gerar todas as métricas: {str(e)}")
                
                else:  # clustering
                    st.markdown("**Modelo de Clustering Criado:**")
                    st.write(f"Algoritmo: K-Means")
                    st.write(f"Número de Clusters: 3")
                    
                    # Mostrar distribuição dos clusters
                    cluster_counts = clustered_data["Cluster"].value_counts().sort_index()
                    
                    fig_clusters = px.bar(
                        x=cluster_counts.index,
                        y=cluster_counts.values,
                        title="Distribuição dos Clusters",
                        labels={"x": "Cluster", "y": "Número de Pontos"}
                    )
                    st.plotly_chart(fig_clusters, use_container_width=True)
                    
                    # Visualização 2D dos clusters (usando as duas primeiras features numéricas)
                    numeric_features = features_data.select_dtypes(include=[np.number]).columns[:2]
                    if len(numeric_features) >= 2:
                        fig_scatter = px.scatter(
                            clustered_data,
                            x=numeric_features[0],
                            y=numeric_features[1],
                            color="Cluster",
                            title=f"Clusters - {numeric_features[0]} vs {numeric_features[1]}"
                        )
                        st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Botão para próxima etapa
                if st.button("➡️ Fazer Previsões", type="secondary"):
                    st.success("✅ Modelo treinado! Vá para a seção \'Previsões\' no menu lateral.")
                
            except Exception as e:
                st.error(f"❌ Erro durante o treinamento: {str(e)}")
                st.error("Verifique se os dados estão no formato correto e tente novamente.")
    
    # Mostrar informações sobre o modelo atual se já treinado
    elif st.session_state.model is not None:
        st.success("✅ Modelo já treinado e pronto para uso!")
        st.info("💡 Você pode retreinar o modelo clicando no botão acima ou ir para a seção de previsões.")

def show_prediction_page():
    """Página para fazer previsões com novos dados"""
    st.markdown("<h2 class=\"section-header\">🔮 Previsões com Novos Dados</h2>", unsafe_allow_html=True)
    
    if st.session_state.model is None:
        st.warning("⚠️ Nenhum modelo treinado. Por favor, treine um modelo primeiro na seção \'Treinamento e Avaliação\'.")
        return
    
    if st.session_state.selected_features is None:
        st.warning("⚠️ Configure o modelo primeiro na seção \'Configuração do Modelo\'.")
        return
    
    # Informações sobre o modelo atual
    st.markdown("<h3 class=\"section-header\">📋 Modelo Atual</h3>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Tipo de Tarefa", st.session_state.task_type.title())
    
    with col2:
        st.metric("Algoritmo", type(st.session_state.model).__name__)
    
    with col3:
        st.metric("Features Necessárias", len(st.session_state.selected_features))
    
    # Opções para entrada de dados
    st.markdown("<h3 class=\"section-header\">📝 Entrada de Dados para Previsão</h3>", unsafe_allow_html=True)
    
    input_method = st.radio(
        "Como você gostaria de inserir os dados?",
        ["📝 Entrada Manual", "📁 Upload de Arquivo CSV"]
    )
    
    if input_method == "📝 Entrada Manual":
        st.markdown("**Insira os valores para cada feature:**")
        
        # Criar formulário para entrada manual
        with st.form("prediction_form"):
            input_data = {}
            
            # Organizar em colunas para melhor layout
            num_cols = 3
            cols = st.columns(num_cols)
            
            for i, feature in enumerate(st.session_state.selected_features):
                col_idx = i % num_cols
                
                with cols[col_idx]:
                    # Verificar tipo da feature no dataset original
                    if st.session_state.df[feature].dtype in ["int64", "float64"]:
                        # Feature numérica
                        min_val = float(st.session_state.df[feature].min())
                        max_val = float(st.session_state.df[feature].max())
                        mean_val = float(st.session_state.df[feature].mean())
                        
                        input_data[feature] = st.number_input(
                            f"{feature}",
                            min_value=min_val,
                            max_value=max_val,
                            value=mean_val,
                            help=f"Valor entre {min_val:.2f} e {max_val:.2f}"
                        )
                    else:
                        # Feature categórica
                        unique_values = st.session_state.df[feature].unique()
                        unique_values = [val for val in unique_values if pd.notna(val)]
                        
                        input_data[feature] = st.selectbox(
                            f"{feature}",
                            unique_values,
                            help=f"Selecione um valor para {feature}"
                        )
            
            # Botão para fazer previsão
            submitted = st.form_submit_button("🔮 Fazer Previsão", type="primary")
            
            if submitted:
                try:
                    # Criar DataFrame com os dados de entrada
                    input_df = pd.DataFrame([input_data])
                    
                    # Fazer previsão
                    if st.session_state.task_type in ["classification", "regression"]:
                        prediction = st.session_state.model.predict(input_df)[0]
                        
                        # Mostrar resultado
                        st.markdown("<h3 class=\"section-header\">🎯 Resultado da Previsão</h3>", unsafe_allow_html=True)
                        
                        if st.session_state.task_type == "classification":
                            st.success(f"**Classe Predita:** {prediction}")
                            
                            # Tentar obter probabilidades se disponível
                            try:
                                probabilities = st.session_state.model.predict_proba(input_df)[0]
                                classes = st.session_state.model.classes_
                                
                                prob_df = pd.DataFrame({
                                    "Classe": classes,
                                    "Probabilidade": probabilities
                                }).sort_values("Probabilidade", ascending=False)
                                
                                st.markdown("**Probabilidades por Classe:**")
                                
                                # Gráfico de barras das probabilidades
                                fig_prob = px.bar(
                                    prob_df,
                                    x="Classe",
                                    y="Probabilidade",
                                    title="Probabilidades de Classificação"
                                )
                                st.plotly_chart(fig_prob, use_container_width=True)
                                
                                # Tabela das probabilidades
                                prob_df["Probabilidade"] = prob_df["Probabilidade"].apply(lambda x: f"{x:.3f}")
                                st.dataframe(prob_df, use_container_width=True)
                                
                            except:
                                st.info("ℹ️ Probabilidades não disponíveis para este modelo.")
                        
                        else:  # regression
                            st.success(f"**Valor Predito:** {prediction:.3f}")
                            
                            # Mostrar contexto da previsão
                            target_col = st.session_state.target_column
                            target_min = st.session_state.df[target_col].min()
                            target_max = st.session_state.df[target_col].max()
                            target_mean = st.session_state.df[target_col].mean()
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Predição", f"{prediction:.3f}")
                            with col2:
                                st.metric("Mínimo no Dataset", f"{target_min:.3f}")
                            with col3:
                                st.metric("Máximo no Dataset", f"{target_max:.3f}")
                            with col4:
                                st.metric("Média no Dataset", f"{target_mean:.3f}")
                    
                    else:  # clustering
                        # Para clustering, atribuir cluster
                        from pycaret.clustering import assign_model
                        
                        clustered_input = assign_model(st.session_state.model, data=input_df)
                        cluster = clustered_input["Cluster"].iloc[0]
                        
                        st.success(f"**Cluster Atribuído:** {cluster}")
                        
                        # Mostrar informações sobre o cluster
                        if hasattr(st.session_state, "clustered_data"):
                            cluster_data = st.session_state.clustered_data
                            cluster_info = cluster_data[cluster_data["Cluster"] == cluster]
                            
                            st.markdown(f"**Informações sobre o Cluster {cluster}:**")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("Pontos no Cluster", len(cluster_info))
                            
                            with col2:
                                total_points = len(cluster_data)
                                percentage = (len(cluster_info) / total_points) * 100
                                st.metric("% do Total", f"{percentage:.1f}%")
                    
                    # Mostrar dados de entrada
                    st.markdown("<h3 class=\"section-header\">📊 Dados de Entrada</h3>", unsafe_allow_html=True)
                    st.dataframe(input_df, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"❌ Erro ao fazer previsão: {str(e)}")
    
    else:  # Upload de arquivo CSV
        st.markdown("**Faça upload de um arquivo CSV com novos dados:**")
        
        uploaded_file = st.file_uploader(
            "Escolha um arquivo CSV",
            type=["csv"],
            help="O arquivo deve conter as mesmas colunas usadas no treinamento"
        )
        
        if uploaded_file is not None:
            try:
                # Ler arquivo
                new_data = pd.read_csv(uploaded_file)
                
                st.markdown("**Preview dos dados carregados:**")
                st.dataframe(new_data.head(), use_container_width=True)
                
                # Verificar se as colunas necessárias estão presentes
                missing_features = [feat for feat in st.session_state.selected_features if feat not in new_data.columns]
                
                if missing_features:
                    st.error(f"❌ Colunas ausentes no arquivo: {missing_features}")
                    st.info("💡 O arquivo deve conter todas as features usadas no treinamento.")
                else:
                    # Selecionar apenas as features necessárias
                    prediction_data = new_data[st.session_state.selected_features].copy()
                    
                    # Verificar valores nulos
                    null_counts = prediction_data.isnull().sum()
                    if null_counts.sum() > 0:
                        st.warning("⚠️ Dados com valores nulos encontrados. Eles serão removidos.")
                        prediction_data = prediction_data.dropna()
                    
                    if len(prediction_data) > 0:
                        if st.button("🔮 Fazer Previsões em Lote", type="primary"):
                            try:
                                with st.spinner("🔄 Fazendo previsões..."):
                                    
                                    if st.session_state.task_type in ["classification", "regression"]:
                                        predictions = st.session_state.model.predict(prediction_data)
                                        
                                        # Adicionar previsões ao DataFrame
                                        result_df = prediction_data.copy()
                                        
                                        if st.session_state.task_type == "classification":
                                            result_df["Classe_Predita"] = predictions
                                            
                                            # Tentar obter probabilidades
                                            try:
                                                probabilities = st.session_state.model.predict_proba(prediction_data)
                                                classes = st.session_state.model.classes_
                                                
                                                for i, class_name in enumerate(classes):
                                                    result_df[f"Prob_{class_name}"] = probabilities[:, i]
                                            except:
                                                pass
                                        
                                        else:  # regression
                                            result_df["Valor_Predito"] = predictions
                                    
                                    else:  # clustering
                                        from pycaret.clustering import assign_model
                                        
                                        clustered_result = assign_model(st.session_state.model, data=prediction_data)
                                        result_df = clustered_result
                                    
                                    # Mostrar resultados
                                    st.markdown("<h3 class=\"section-header\">🎯 Resultados das Previsões</h3>", unsafe_allow_html=True)
                                    
                                    st.dataframe(result_df, use_container_width=True)
                                    
                                    # Estatísticas das previsões
                                    if st.session_state.task_type == "classification":
                                        pred_counts = result_df["Classe_Predita"].value_counts()
                                        
                                        fig_pred_dist = px.bar(
                                            x=pred_counts.index,
                                            y=pred_counts.values,
                                            title="Distribuição das Previsões",
                                            labels={"x": "Classe Predita", "y": "Frequência"}
                                        )
                                        st.plotly_chart(fig_pred_dist, use_container_width=True)
                                    
                                    elif st.session_state.task_type == "regression":
                                        fig_pred_hist = px.histogram(
                                            result_df,
                                            x="Valor_Predito",
                                            title="Distribuição dos Valores Preditos",
                                            nbins=30
                                        )
                                        st.plotly_chart(fig_pred_hist, use_container_width=True)
                                    
                                    else:  # clustering
                                        cluster_counts = result_df["Cluster"].value_counts().sort_index()
                                        
                                        fig_cluster_dist = px.bar(
                                            x=cluster_counts.index,
                                            y=cluster_counts.values,
                                            title="Distribuição dos Clusters",
                                            labels={"x": "Cluster", "y": "Frequência"}
                                        )
                                        st.plotly_chart(fig_cluster_dist, use_container_width=True)
                                    
                                    # Opção para download dos resultados
                                    csv_result = result_df.to_csv(index=False)
                                    st.download_button(
                                        label="📥 Download dos Resultados (CSV)",
                                        data=csv_result,
                                        file_name="predicoes_resultado.csv",
                                        mime="text/csv"
                                    )
                                    
                                    st.success(f"✅ Previsões concluídas para {len(result_df)} registros!")
                            
                            except Exception as e:
                                st.error(f"❌ Erro ao fazer previsões: {str(e)}")
                    else:
                        st.error("❌ Nenhum dado válido encontrado após remoção de valores nulos.")
            
            except Exception as e:
                st.error(f"❌ Erro ao carregar arquivo: {str(e)}")
    
    # Informações sobre as features necessárias
    st.markdown("<h3 class=\"section-header\">📋 Features Necessárias</h3>", unsafe_allow_html=True)
    st.info("💡 Para fazer previsões, você precisa fornecer valores para as seguintes features:")
    
    features_info = []
    for feature in st.session_state.selected_features:
        feature_type = "Numérica" if st.session_state.df[feature].dtype in ["int64", "float64"] else "Categórica"
        
        if feature_type == "Numérica":
            min_val = st.session_state.df[feature].min()
            max_val = st.session_state.df[feature].max()
            info = f"Valor entre {min_val:.2f} e {max_val:.2f}"
        else:
            unique_vals = st.session_state.df[feature].nunique()
            info = f"{unique_vals} valores únicos"
        
        features_info.append({
            "Feature": feature,
            "Tipo": feature_type,
            "Informação": info
        })
    
    features_df = pd.DataFrame(features_info)
    st.dataframe(features_df, use_container_width=True)

def main():
    # Título principal
    st.markdown("<h1 class=\"main-header\">🤖 ML Studio - Análise e Modelagem</h1>", unsafe_allow_html=True)
    
    # Sidebar para navegação
    st.sidebar.title("📋 Menu de Navegação")
    
    # Opções do menu
    menu_options = [
        "🏠 Início",
        "📊 Upload e Análise de Dados",
        "🔍 Análise Exploratória",
        "⚙️ Configuração do Modelo",
        "🎯 Treinamento e Avaliação",
        "🔮 Previsões"
    ]
    
    selected_option = st.sidebar.selectbox("Selecione uma opção:", menu_options)
    
    # Inicializar session state
    if "df" not in st.session_state:
        st.session_state.df = None
    if "model" not in st.session_state:
        st.session_state.model = None
    if "target_column" not in st.session_state:
        st.session_state.target_column = None
    if "task_type" not in st.session_state:
        st.session_state.task_type = None
    if "selected_features" not in st.session_state:
        st.session_state.selected_features = None
    
    # Roteamento baseado na seleção do menu
    if selected_option == "🏠 Início":
        show_home_page()
    elif selected_option == "📊 Upload e Análise de Dados":
        show_data_upload_page()
    elif selected_option == "🔍 Análise Exploratória":
        show_exploratory_analysis_page()
    elif selected_option == "⚙️ Configuração do Modelo":
        show_model_configuration_page()
    elif selected_option == "🎯 Treinamento e Avaliação":
        show_training_page()
    elif selected_option == "🔮 Previsões":
        show_prediction_page()

if __name__ == "__main__":
    main()

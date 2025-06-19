import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Adicionar o diretório src ao path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
# sys.path.append(os.path.join(PROJECT_ROOT, 'src')) # Removido se não for estritamente necessário aqui, já que estamos em pages

st.title("📊 Métricas e Avaliação dos Modelos")
st.markdown("---")

# Sidebar para configurações
st.sidebar.header("⚙️ Configurações de Avaliação")

# Seleção do ativo
asset_options = {
    "Apple (AAPL)": "aapl",
    "Bitcoin (BTC-USD)": "btc"
}

selected_asset_name = st.sidebar.selectbox(
    "Selecione o Ativo:",
    options=list(asset_options.keys()),
    index=0
)
selected_asset_code = asset_options[selected_asset_name]

# Seleção do modelo
model_options = {
    "LSTM": "lstm",
    "GRU": "gru",
    "SimpleRNN": "simplernn"
}

selected_model_type = st.sidebar.selectbox(
    "Selecione o Modelo:",
    options=list(model_options.keys()),
    index=0
)
selected_model_code = model_options[selected_model_type]

# Parâmetros de avaliação
st.sidebar.subheader("Parâmetros de Teste")
test_size = st.sidebar.slider(
    "Tamanho do Conjunto de Teste (%):",
    min_value=10,
    max_value=30,
    value=20,
    step=5
)

forecast_horizon = st.sidebar.selectbox(
    "Horizonte de Previsão (dias):",
    options=[1, 3, 5, 7, 14],
    index=2
)

@st.cache_data
def load_processed_data(asset_name):
    """Carrega os dados processados do ativo selecionado."""
    # Corrigir o caminho para ser absoluto
    file_path = os.path.join(PROJECT_ROOT, "data", "processed", f"{asset_name}_processed.csv")
    try:
        df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
        
        # Garantir que temos as colunas básicas necessárias
        if 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
        
        # Criar features técnicas necessárias para compatibilidade com o modelo
        if 'MA_5' not in df.columns:
            df['MA_5'] = df['Close'].rolling(window=5, min_periods=1).mean()
        if 'MA_20' not in df.columns:
            df['MA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
        if 'Volume_MA' not in df.columns:
            df['Volume_MA'] = df['Volume'].rolling(window=10, min_periods=1).mean()
        
        # Remover valores NaN
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        return df
    except FileNotFoundError:
        st.error(f"Arquivo de dados processados não encontrado para {asset_name.upper()} em {file_path}")
        st.info(f"Verifique se o arquivo existe em: {os.path.abspath(file_path)}")
        return None

@st.cache_resource
def load_model_and_scaler(asset_name, model_type):
    """Carrega o modelo treinado e o scaler."""
    # Corrigir os caminhos para serem absolutos
    model_path = os.path.join(PROJECT_ROOT, "models", "models", f"{asset_name}_{model_type}_best.h5")
    scaler_path = os.path.join(PROJECT_ROOT, "models", "models", f"{asset_name}_scaler.joblib")
    
    model = None
    scaler = None
    
    try:
        model = load_model(model_path)
    except Exception as e:
        st.warning(f"Modelo não encontrado: {model_path}. Erro: {e}")
    
    try:
        scaler = joblib.load(scaler_path)
    except Exception as e:
        st.warning(f"Scaler não encontrado: {scaler_path}. Erro: {e}")
    
    return model, scaler

def prepare_sequences(data, sequence_length=60):
    """Prepara sequências para avaliação do modelo."""
    model_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_5', 'MA_20', 'Volume_MA']
    
    # Verificar se todas as features necessárias existem
    missing_features = [col for col in model_features if col not in data.columns]
    if missing_features:
        st.error(f"Features faltantes: {missing_features}")
        return None, None
    
    # Selecionar apenas as features do modelo
    feature_data = data[model_features].copy()
    
    X, y = [], []
    for i in range(sequence_length, len(feature_data)):
        X.append(feature_data.iloc[i-sequence_length:i].values)
        y.append(feature_data.iloc[i]['Close'])
    
    return np.array(X), np.array(y)

def evaluate_model(model, scaler, X_test, y_test):
    """Avalia o modelo e calcula métricas."""
    if model is None or scaler is None:
        return None
    
    # Normalizar dados de teste
    X_test_scaled = np.array([scaler.transform(seq) for seq in X_test])
    
    # Fazer previsões
    predictions_scaled = model.predict(X_test_scaled, verbose=0)
    
    # Desnormalizar previsões
    predictions = []
    y_true = []
    
    for i, pred_scaled in enumerate(predictions_scaled):
        # Criar array dummy para desnormalização
        dummy_pred = np.zeros((1, scaler.n_features_in_))
        dummy_pred[0, 3] = pred_scaled[0] if len(pred_scaled.shape) > 1 else pred_scaled
        pred_denorm = scaler.inverse_transform(dummy_pred)[0, 3]
        predictions.append(pred_denorm)
        
        # Desnormalizar valor real
        dummy_true = np.zeros((1, scaler.n_features_in_))
        dummy_true[0, 3] = (y_test[i] - scaler.data_min_[3]) / (scaler.data_max_[3] - scaler.data_min_[3])
        true_denorm = scaler.inverse_transform(dummy_true)[0, 3]
        y_true.append(true_denorm)
    
    predictions = np.array(predictions)
    y_true = np.array(y_true)
    
    # Calcular métricas
    mse = mean_squared_error(y_true, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, predictions)
    r2 = r2_score(y_true, predictions)
    
    # Métricas específicas para previsão de preços
    mape = np.mean(np.abs((y_true - predictions) / y_true)) * 100
    directional_accuracy = np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(predictions))) * 100
    
    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2,
        "MAPE (%)": mape,
        "Acurácia Direcional (%)": directional_accuracy
    }
    
    return predictions, y_true, metrics

def plot_predictions_vs_actual(predictions, y_true, dates, asset_name, model_type):
    """Plota previsões vs valores reais."""
    fig = go.Figure()
    
    # Valores reais
    fig.add_trace(go.Scatter(
        x=dates,
        y=y_true,
        mode='lines',
        name='Valores Reais',
        line=dict(color='blue', width=2)
    ))
    
    # Previsões
    fig.add_trace(go.Scatter(
        x=dates,
        y=predictions,
        mode='lines',
        name='Previsões',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title=f'Previsões vs Valores Reais - {asset_name} ({model_type})',
        xaxis_title='Data',
        yaxis_title='Preço (USD)',
        height=500,
        hovermode='x unified'
    )
    
    return fig

def plot_residuals(predictions, y_true, dates):
    """Plota análise de resíduos."""
    residuals = y_true - predictions
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Resíduos ao Longo do Tempo', 'Distribuição dos Resíduos', 
                       'Q-Q Plot', 'Resíduos vs Previsões'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Resíduos ao longo do tempo
    fig.add_trace(
        go.Scatter(x=dates, y=residuals, mode='lines', name='Resíduos'),
        row=1, col=1
    )
    
    # Distribuição dos resíduos
    fig.add_trace(
        go.Histogram(x=residuals, name='Distribuição', nbinsx=30),
        row=1, col=2
    )
    
    # Q-Q Plot (aproximado)
    from scipy import stats
    qq_theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
    qq_sample = np.sort(residuals)
    fig.add_trace(
        go.Scatter(x=qq_theoretical, y=qq_sample, mode='markers', name='Q-Q Plot'),
        row=2, col=1
    )
    
    # Resíduos vs Previsões
    fig.add_trace(
        go.Scatter(x=predictions, y=residuals, mode='markers', name='Resíduos vs Pred'),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    
    return fig

# Interface principal
st.header(f"📈 Avaliação do Modelo {selected_model_type}")
st.subheader(f"Ativo: {selected_asset_name}")

# Carregar dados
with st.spinner(f"Carregando dados de {selected_asset_name}..."):
    df = load_processed_data(selected_asset_code)

if df is not None:
    # Carregar modelo e scaler
    with st.spinner(f"Carregando modelo {selected_model_type}..."):
        model, scaler = load_model_and_scaler(selected_asset_code, selected_model_code)
    
    if model is not None and scaler is not None:
        # Preparar dados de teste
        with st.spinner("Preparando dados de teste..."):
            # Dividir dados
            split_idx = int(len(df) * (1 - test_size/100))
            test_data = df.iloc[split_idx:]
            
            # Preparar sequências
            X_test, y_test = prepare_sequences(test_data)
            
            if X_test is not None and y_test is not None:
                # Avaliar modelo
                with st.spinner("Avaliando modelo..."):
                    predictions, y_true, metrics = evaluate_model(model, scaler, X_test, y_test)
                
                if predictions is not None:
                    # Exibir métricas principais
                    st.subheader("📊 Métricas de Performance")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("RMSE", f"{metrics['RMSE']:.4f}")
                        st.metric("R²", f"{metrics['R²']:.4f}")
                    
                    with col2:
                        st.metric("MAE", f"{metrics['MAE']:.4f}")
                        st.metric("MAPE (%)", f"{metrics['MAPE (%)']:.2f}%")
                    
                    with col3:
                        st.metric("MSE", f"{metrics['MSE']:.4f}")
                        st.metric("Acurácia Direcional", f"{metrics['Acurácia Direcional (%)']:.2f}%")
                    
                    # Gráfico de previsões vs valores reais
                    st.subheader("📈 Previsões vs Valores Reais")
                    
                    test_dates = test_data.index[60:]  # Ajustar para sequências
                    fig_pred = plot_predictions_vs_actual(
                        predictions, y_true, test_dates, 
                        selected_asset_name, selected_model_type
                    )
                    st.plotly_chart(fig_pred, use_container_width=True)
                    
                    # Análise de resíduos
                    st.subheader("📉 Análise de Resíduos")
                    
                    fig_residuals = plot_residuals(predictions, y_true, test_dates)
                    st.plotly_chart(fig_residuals, use_container_width=True)
                    
                    # Tabela detalhada de métricas
                    st.subheader("📋 Métricas Detalhadas")
                    
                    metrics_df = pd.DataFrame({
                        "Métrica": list(metrics.keys()),
                        "Valor": [f"{v:.4f}" if isinstance(v, float) else str(v) for v in metrics.values()],
                        "Descrição": [
                            "Erro Quadrático Médio",
                            "Raiz do Erro Quadrático Médio",
                            "Erro Absoluto Médio",
                            "Coeficiente de Determinação",
                            "Erro Percentual Absoluto Médio",
                            "% de acerto na direção do movimento"
                        ]
                    })
                    
                    st.dataframe(metrics_df, use_container_width=True)
                    
                    # Estatísticas dos erros
                    st.subheader("📊 Estatísticas dos Erros")
                    
                    errors = y_true - predictions
                    error_stats = {
                        "Erro Médio": np.mean(errors),
                        "Desvio Padrão dos Erros": np.std(errors),
                        "Erro Mínimo": np.min(errors),
                        "Erro Máximo": np.max(errors),
                        "Percentil 25": np.percentile(errors, 25),
                        "Mediana dos Erros": np.median(errors),
                        "Percentil 75": np.percentile(errors, 75)
                    }
                    
                    error_df = pd.DataFrame({
                        "Estatística": list(error_stats.keys()),
                        "Valor": [f"{v:.4f}" for v in error_stats.values()]
                    })
                    
                    st.dataframe(error_df, use_container_width=True)
                    
                else:
                    st.error("Erro ao avaliar o modelo.")
            else:
                st.error("Erro ao preparar sequências de teste.")
    else:
        st.warning(f"Modelo {selected_model_type} ou scaler não disponível para {selected_asset_name}.")
        st.info("💡 **Dica:** Execute o treinamento do modelo primeiro.")
else:
    st.error("❌ Não foi possível carregar os dados.")
    st.info("💡 **Dica:** Verifique se os arquivos de dados processados existem.")

# Informações sobre métricas
with st.expander("ℹ️ Sobre as Métricas"):
    st.markdown("""
    ### Métricas de Avaliação:
    
    **MSE (Mean Squared Error):**
    - Média dos quadrados dos erros
    - Penaliza mais os erros grandes
    - Valores menores são melhores
    
    **RMSE (Root Mean Squared Error):**
    - Raiz quadrada do MSE
    - Mesma unidade dos dados originais
    - Interpretação mais intuitiva
    
    **MAE (Mean Absolute Error):**
    - Média dos valores absolutos dos erros
    - Menos sensível a outliers que RMSE
    - Valores menores são melhores
    
    **R² (Coeficiente de Determinação):**
    - Proporção da variância explicada pelo modelo
    - Varia de 0 a 1 (1 = perfeito)
    - Indica qualidade do ajuste
    
    **MAPE (Mean Absolute Percentage Error):**
    - Erro percentual médio
    - Facilita comparação entre diferentes escalas
    - Valores menores são melhores
    
    **Acurácia Direcional:**
    - % de acerto na direção do movimento (alta/baixa)
    - Importante para estratégias de trading
    - Valores maiores são melhores
    """)

st.markdown("---")
st.markdown("**MIT-510 - Módulo de Avaliação de Modelos**")
st.markdown("Análise detalhada da performance dos modelos de Machine Learning")
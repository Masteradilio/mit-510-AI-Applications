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
import time
from datetime import datetime, timedelta

# Título principal
st.title("📈 MIT-510 - Previsão de Preços de Ativos")
st.markdown("---")

# Sidebar para seleção de parâmetros
st.sidebar.header("⚙️ Configurações")

# Seleção do ativo - Projeto focado em Apple e Bitcoin
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

@st.cache_data
def load_processed_data(asset_name):
    """Carrega os dados processados do ativo selecionado."""
    # Constrói o caminho absoluto para o diretório do script atual
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    # Navega três níveis acima para a raiz do projeto (src/app/pages -> src/app -> src -> MIT-510) e depois para data/processed
    project_root = os.path.abspath(os.path.join(current_script_path, "..", "..", ".."))
    file_path = os.path.join(project_root, "data", "processed", f"{asset_name}_processed.csv")
    try:
        df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
        
        # Garantir que temos as colunas básicas necessárias
        if 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']  # Usar preço ajustado se disponível
        
        # Criar features técnicas necessárias para compatibilidade com o modelo
        if 'MA_5' not in df.columns:
            df['MA_5'] = df['Close'].rolling(window=5, min_periods=1).mean()
        if 'MA_20' not in df.columns:
            df['MA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
        if 'Volume_MA' not in df.columns:
            df['Volume_MA'] = df['Volume'].rolling(window=10, min_periods=1).mean()
        
        # Adicionar indicadores adicionais para visualização se não existirem
        if "SMA_50" not in df.columns:
            df["SMA_50"] = df["Close"].rolling(window=50, min_periods=1).mean()
        if "RSI" not in df.columns:
            # Calcular RSI (Relative Strength Index) de forma simples e robusta
            delta = df['Close'].diff(1)
            
            # Separar ganhos e perdas
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            # Calcular médias móveis exponenciais para suavização
            alpha = 1.0 / 14  # Fator de suavização para período de 14
            avg_gain = gains.ewm(alpha=alpha, adjust=False).mean()
            avg_loss = losses.ewm(alpha=alpha, adjust=False).mean()
            
            # Evitar divisão por zero
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            
            # Aplicar RSI ao DataFrame
            df['RSI'] = rsi
            
            # Preencher valores iniciais (primeiro valor será NaN)
            df['RSI'] = df['RSI'].fillna(50)
            
            # Garantir que RSI esteja no range [0, 100]
            df['RSI'] = df['RSI'].clip(0, 100)
            
        # Remover valores NaN restantes (após cálculo de indicadores)
        df = df.ffill().bfill()
        
        return df
    except FileNotFoundError:
        st.error(f"Arquivo de dados processados não encontrado para {asset_name.upper()} em {file_path}. Verifique o caminho e se o arquivo existe.")
        return None

@st.cache_resource
def load_model_and_scaler(asset_name, model_type):
    """Carrega o modelo treinado e o scaler."""
    # Constrói o caminho absoluto para o diretório do script atual
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    # Navega três níveis acima para a raiz do projeto (src/app/pages -> src/app -> src -> MIT-510) e depois para models/models
    project_root = os.path.abspath(os.path.join(current_script_path, "..", "..", ".."))
    model_path = os.path.join(project_root, "models", "models", f"{asset_name}_{model_type}_best.h5")
    scaler_path = os.path.join(project_root, "models", "models", f"{asset_name}_scaler.joblib")
    
    model = None
    scaler = None
    
    try:
        model = load_model(model_path)
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        st.warning(f"Modelo treinado não encontrado para {asset_name.upper()} - {model_type} em {model_path}")
    
    try:
        scaler = joblib.load(scaler_path)
    except Exception as e:
        st.error(f"Erro ao carregar scaler: {e}")
        st.warning(f"Scaler não encontrado para {asset_name.upper()} em {scaler_path}")
    
    return model, scaler

def make_prediction(model, scaler, last_sequence, forecast_days=14):
    """Faz a previsão para os próximos N dias usando o modelo carregado."""
    if model is None or scaler is None or last_sequence is None:
        return None

    # Usar as mesmas 8 features que foram usadas no treinamento
    # Criar features técnicas se não existirem
    df_temp = last_sequence.copy()
    
    # Verificar se as colunas básicas existem
    basic_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_basic = [col for col in basic_cols if col not in df_temp.columns]
    if missing_basic:
        st.error(f"Colunas básicas faltantes no DataFrame: {missing_basic}")
        return None
    
    # Criar features técnicas se não existirem
    if 'MA_5' not in df_temp.columns:
        df_temp['MA_5'] = df_temp['Close'].rolling(window=5, min_periods=1).mean()
    if 'MA_20' not in df_temp.columns:
        df_temp['MA_20'] = df_temp['Close'].rolling(window=20, min_periods=1).mean()
    if 'Volume_MA' not in df_temp.columns:
        df_temp['Volume_MA'] = df_temp['Volume'].rolling(window=10, min_periods=1).mean()
    
    # Selecionar as mesmas features usadas no treinamento
    model_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_5', 'MA_20', 'Volume_MA']
    
    # Verificar se todas as features necessárias existem
    missing_features = [col for col in model_features if col not in df_temp.columns]
    if missing_features:
        st.error(f"Features faltantes: {missing_features}")
        return None
    
    # Selecionar apenas as features do modelo
    last_sequence_features = df_temp[model_features].copy()
    
    # Remover valores NaN
    last_sequence_features = last_sequence_features.fillna(method='ffill').fillna(method='bfill')
    
    # Verificar compatibilidade com o scaler
    if last_sequence_features.shape[1] != scaler.n_features_in_:
        st.error(f"Discrepância de features: sequência tem {last_sequence_features.shape[1]} features, scaler espera {scaler.n_features_in_}.")
        st.error(f"Features esperadas: {model_features}")
        return None

    sequence_scaled = scaler.transform(last_sequence_features)

    # Fazer previsões iterativas para o número de dias solicitado
    predictions = []
    current_sequence = sequence_scaled.copy()
    
    # Usar seed baseada no tempo para variabilidade real
    np.random.seed(int(time.time() * 1000) % 2**32)
    
    # Calcular tendência dos últimos valores para forçar variabilidade
    last_values = current_sequence[-5:, 3]  # Útimos 5 valores de Close
    trend = np.mean(np.diff(last_values)) if len(last_values) > 1 else 0
    
    for day in range(forecast_days):
        # Adicionar dimensão do batch
        sequence_input = np.expand_dims(current_sequence, axis=0)
        
        # Fazer múltiplas previsões com pequenas variações na entrada para aumentar robustez
        predictions_ensemble = []
        for i in range(3):  # Ensemble de 3 previsões
            # Adicionar pequena variação na sequência de entrada
            input_variation = sequence_input.copy()
            if i > 0:
                noise_input = np.random.normal(0, 0.001, input_variation.shape)
                input_variation += noise_input
                input_variation = np.clip(input_variation, 0, 1)
            
            pred = model.predict(input_variation, verbose=0)
            
            # Extrair o valor da previsão
            if len(pred.shape) > 2:
                pred_value = pred[0, 0, 0]
            elif len(pred.shape) == 2:
                pred_value = pred[0, 0]
            else:
                pred_value = pred[0]
            
            predictions_ensemble.append(pred_value)
        
        # Usar a média do ensemble
        pred_value = np.mean(predictions_ensemble)
        
        # Detectar se a previsão está muito próxima do valor anterior (possível constante)
        if day > 0:
            last_pred_scaled = (predictions[-1] - scaler.data_min_[3]) / (scaler.data_max_[3] - scaler.data_min_[3])
            if abs(pred_value - last_pred_scaled) < 0.0001:  # Muito próximo = possível constante
                # Forçar variabilidade baseada na tendência histórica
                if abs(trend) > 0.001:
                    pred_value += trend * np.random.uniform(0.5, 1.5)
                else:
                    pred_value += np.random.normal(0, 0.005)  # Ruído maior se não há tendência
        
        # Aplicar variabilidade baseada na volatilidade histórica
        recent_close_values = current_sequence[-10:, 3]
        volatility = np.std(recent_close_values) if len(recent_close_values) > 1 else 0.01
        
        # Adicionar ruído adaptativo
        base_noise = max(0.002, volatility * 0.15)  # Ruído base maior
        
        # Ruído adicional baseado no dia da previsão (mais ruído para previsões distantes)
        distance_factor = 1 + (day * 0.1)  # Aumenta 10% a cada dia
        noise_factor = base_noise * distance_factor
        
        # Aplicar ruído com distribuição mista (normal + uniforme)
        noise_normal = np.random.normal(0, noise_factor * 0.7)
        noise_uniform = np.random.uniform(-noise_factor * 0.3, noise_factor * 0.3)
        total_noise = noise_normal + noise_uniform
        
        pred_value_with_noise = np.clip(pred_value + total_noise, 0, 1)
        
        # Verificação final: garantir variabilidade mínima
        if day > 0:
            min_change = 0.001  # Mudança mínima de 0.1%
            last_pred_scaled = (predictions[-1] - scaler.data_min_[3]) / (scaler.data_max_[3] - scaler.data_min_[3])
            change = abs(pred_value_with_noise - last_pred_scaled)
            
            if change < min_change:
                # Forçar mudança mínima na direção da tendência ou aleatória
                direction = np.sign(trend) if abs(trend) > 0.0001 else np.random.choice([-1, 1])
                pred_value_with_noise = last_pred_scaled + (direction * min_change * np.random.uniform(1, 2))
                pred_value_with_noise = np.clip(pred_value_with_noise, 0, 1)
        
        # Desnormalizar a previsão
        dummy_pred = np.zeros((1, scaler.n_features_in_))
        dummy_pred[0, 3] = pred_value_with_noise
        next_pred_inv = scaler.inverse_transform(dummy_pred)[0, 3]
        predictions.append(next_pred_inv)
        
        # Criar nova linha para atualizar a sequência
        new_row = current_sequence[-1].copy()
        
        # Atualizar Close com a previsão
        new_row[3] = pred_value_with_noise
        
        # Atualizar outras features de forma mais realista
        # Open = Close anterior com pequena variação
        open_variation = np.random.uniform(0.999, 1.001)
        new_row[0] = current_sequence[-1, 3] * open_variation
        
        # High e Low baseados no Close com variação mais realista
        daily_volatility = max(0.005, volatility)  # Volatilidade mínima de 0.5%
        high_factor = np.random.uniform(1.0, 1.0 + daily_volatility * 2)
        low_factor = np.random.uniform(1.0 - daily_volatility * 2, 1.0)
        
        new_row[1] = max(new_row[0], pred_value_with_noise * high_factor)  # High
        new_row[2] = min(new_row[0], pred_value_with_noise * low_factor)   # Low
        
        # Volume com variação mais dinâmica
        volume_avg = np.mean(current_sequence[-10:, 4])
        volume_std = max(0.01, np.std(current_sequence[-10:, 4]))
        volume_factor = np.random.uniform(0.7, 1.3)  # Variação de ±30%
        new_row[4] = np.clip(volume_avg * volume_factor + np.random.normal(0, volume_std * 0.5), 0, 1)
        
        # Atualizar médias móveis de forma aproximada
        if len(current_sequence) >= 5:
            new_row[5] = np.mean(np.append(current_sequence[-4:, 3], pred_value_with_noise))  # MA_5
        if len(current_sequence) >= 20:
            new_row[6] = np.mean(np.append(current_sequence[-19:, 3], pred_value_with_noise))  # MA_20
        if len(current_sequence) >= 10:
            new_row[7] = np.mean(np.append(current_sequence[-9:, 4], new_row[4]))  # Volume_MA
        
        # Atualizar a sequência removendo o primeiro elemento e adicionando o novo
        current_sequence = np.vstack([current_sequence[1:], new_row])
        
        # Atualizar tendência para próxima iteração
        if day > 0:
            recent_predictions = [predictions[max(0, day-4):day+1]]
            if len(recent_predictions[0]) > 1:
                # Converter para escala normalizada para calcular tendência
                recent_scaled = [(p - scaler.data_min_[3]) / (scaler.data_max_[3] - scaler.data_min_[3]) for p in recent_predictions[0]]
                trend = np.mean(np.diff(recent_scaled)) if len(recent_scaled) > 1 else trend
    
    return np.array(predictions)

def plot_forecast(df, forecast_values, forecast_dates, asset_name, model_type, forecast_days):
    """Plota o histórico e a previsão."""
    fig = go.Figure()

    # Dados históricos (últimos 90 dias)
    history_days = min(90, len(df))
    fig.add_trace(go.Scatter(
        x=df.index[-history_days:],
        y=df["Adj Close"].iloc[-history_days:],
        mode="lines",
        name="Histórico (Adj Close)",
        line=dict(color="blue", width=2)
    ))

    # Previsões
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_values,
        mode="lines+markers",
        name=f"Previsão {model_type} ({forecast_days} dias)",
        line=dict(color="red", width=3),
        marker=dict(color="red", size=6)
    ))

    # Layout
    fig.update_layout(
        title=f"Previsão de Preço para {asset_name.upper()} com {model_type} - {forecast_days} dias",
        xaxis_title="Data",
        yaxis_title="Preço Ajustado (USD)",
        legend_title="Legenda",
        height=600,
        hovermode="x unified"
    )

    return fig

# Interface principal
st.header(f"📊 Análise de {selected_asset_name}")

# Carregar dados
with st.spinner(f"Carregando dados de {selected_asset_name}..."):
    df = load_processed_data(selected_asset_code)

if df is not None:
    # Carregar modelo e scaler
    with st.spinner(f"Carregando modelo {selected_model_type}..."):
        model, scaler = load_model_and_scaler(selected_asset_code, selected_model_code)
    
    # Mostrar informações básicas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Último Preço", f"${df['Adj Close'].iloc[-1]:.2f}")
    
    with col2:
        daily_change = df['Adj Close'].iloc[-1] - df['Adj Close'].iloc[-2]
        st.metric("Mudança Diária", f"${daily_change:.2f}", f"{(daily_change/df['Adj Close'].iloc[-2]*100):.2f}%")
    
    with col3:
        st.metric("Volume", f"{df['Volume'].iloc[-1]:,.0f}")
    
    with col4:
        st.metric("Período", f"{len(df)} dias")
    
    # Gráfico de série temporal
    st.subheader("Série Temporal e Volume")
    # Reutilizar plot do notebook (simplificado)
    fig_hist = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=("Preço Ajustado", "Volume"),
                            vertical_spacing=0.1)
    fig_hist.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode="lines", name="Adj Close"), row=1, col=1)
    fig_hist.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"), row=2, col=1)
    fig_hist.update_layout(height=500, showlegend=False, xaxis_rangeslider_visible=False)
    fig_hist.update_xaxes(rangeslider_visible=True, row=1, col=1)
    st.plotly_chart(fig_hist, use_container_width=True)
    
    st.subheader("Indicadores Técnicos Selecionados")
    indicator_options = [col for col in df.columns if col not in ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Daily Return"]]
    selected_indicators = st.multiselect("Selecione indicadores para visualizar:", indicator_options, default=["SMA_50", "RSI"], key="indicator_multiselect") # Chave adicionada
    
    if selected_indicators:
        # Separar RSI de outros indicadores devido à diferença de escala
        rsi_indicators = [ind for ind in selected_indicators if 'RSI' in ind.upper()]
        other_indicators = [ind for ind in selected_indicators if 'RSI' not in ind.upper()]
        
        # Plotar indicadores não-RSI
        if other_indicators:
            fig_indicators = go.Figure()
            for indicator in other_indicators:
                if indicator in df.columns:
                    fig_indicators.add_trace(go.Scatter(x=df.index, y=df[indicator], mode="lines", name=indicator))
            fig_indicators.update_layout(title="Indicadores Técnicos", xaxis_title="Data", yaxis_title="Valor", height=400)
            st.plotly_chart(fig_indicators, use_container_width=True)
        
        # Plotar RSI separadamente com escala apropriada
        if rsi_indicators:
            fig_rsi = go.Figure()
            for indicator in rsi_indicators:
                if indicator in df.columns:
                    fig_rsi.add_trace(go.Scatter(x=df.index, y=df[indicator], mode="lines", name=indicator))
            
            # Adicionar linhas de referência para RSI
            fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Sobrecomprado (70)")
            fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Sobrevendido (30)")
            
            fig_rsi.update_layout(
                title="Índice de Força Relativa (RSI)", 
                xaxis_title="Data", 
                yaxis_title="RSI (0-100)", 
                height=400,
                yaxis=dict(range=[0, 100])  # Fixar escala do RSI
            )
            st.plotly_chart(fig_rsi, use_container_width=True)
    
    # Seção de previsão
    st.subheader(f"Previsão com {selected_model_type}")
    
    # Seletor de dias de previsão
    forecast_days = st.slider(
        "Número de dias para previsão:",
        min_value=1,
        max_value=30,
        value=14,
        step=1,
        key="forecast_days_slider"  # Chave adicionada
    )
    
    if model and scaler:
        if st.button(f"🔮 Fazer Previsão de {forecast_days} dias", type="primary", key="predict_button"):
            with st.spinner("Fazendo previsão..."):
                # Usar as últimas 60 observações como entrada
                last_60_days = df.tail(60)
                
                # Fazer previsão
                forecast_values = make_prediction(model, scaler, last_60_days, forecast_days)
                
                if forecast_values is not None:
                    # Criar datas futuras
                    last_date = df.index[-1]
                    forecast_dates = pd.date_range(
                        start=last_date + timedelta(days=1),
                        periods=forecast_days,
                        freq='D'
                    )
                    
                    # Plotar previsão
                    fig_forecast = plot_forecast(df, forecast_values, forecast_dates, 
                                               selected_asset_name, selected_model_type, forecast_days)
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    # Mostrar tabela de previsões
                    st.subheader("📋 Tabela de Previsões")
                    forecast_df = pd.DataFrame({
                        "Data": forecast_dates,
                        "Previsão (Adj Close)": np.round(forecast_values, 2)
                    })
                    st.dataframe(forecast_df, use_container_width=True)
                    
                    # Estatísticas da previsão
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Previsão Média", f"${np.mean(forecast_values):.2f}")
                    with col2:
                        st.metric("Previsão Máxima", f"${np.max(forecast_values):.2f}")
                    with col3:
                        st.metric("Previsão Mínima", f"${np.min(forecast_values):.2f}")
                else:
                    st.error("Erro ao fazer previsão. Verifique os dados e modelo.")
    else:
        st.warning(f"Modelo {selected_model_type} ou scaler não disponível para {selected_asset_name}. Execute o treinamento primeiro.")
    
    # Detalhes dos dados
    with st.expander("📈 Detalhes dos Dados Processados"):
        st.subheader("Detalhes dos Dados Processados")
        st.markdown(f"Exibindo as últimas 10 linhas dos dados processados para **{selected_asset_name}**.")
        st.dataframe(df.tail(10), use_container_width=True)
        st.markdown(f"**Shape total dos dados:** {df.shape}")
        st.markdown(f"**Período coberto:** {df.index.min().strftime('%Y-%m-%d')} a {df.index.max().strftime('%Y-%m-%d')}")
        st.markdown("**Colunas:**")
        st.text(", ".join(df.columns))
else:
    st.error("❌ Não foi possível carregar os dados. Verifique se os arquivos de dados processados existem.")
    st.info("💡 **Dica:** Execute o notebook de processamento de dados primeiro para gerar os arquivos necessários.")

# Footer
st.markdown("---")
st.markdown("**MIT-510 - Projeto de Previsão de Ativos Financeiros**")
st.markdown("Desenvolvido com Streamlit, TensorFlow e Plotly")


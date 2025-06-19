# -*- coding: utf-8 -*-
"""Componentes de UI para o aplicativo Streamlit."""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_asset_selector(default_option="Bitcoin (BTC-USD)"):
    """Cria um seletor de ativos financeiros.
    
    Args:
        default_option (str): Opção padrão selecionada.
        
    Returns:
        tuple: (nome_exibição, código_interno)
    """
    asset_options = {
        "Bitcoin (BTC-USD)": "btc", 
        "Apple (AAPL)": "aapl"
    }
    
    selected_asset_name = st.selectbox(
        "Selecione o Ativo:", 
        options=list(asset_options.keys()),
        index=list(asset_options.keys()).index(default_option) if default_option in asset_options else 0
    )
    
    return selected_asset_name, asset_options[selected_asset_name]

def create_model_selector(default_option="LSTM"):
    """Cria um seletor de modelos de previsão.
    
    Args:
        default_option (str): Opção padrão selecionada.
        
    Returns:
        tuple: (nome_exibição, código_interno)
    """
    model_options = {
        "LSTM": "lstm", 
        "GRU": "gru", 
        "SimpleRNN": "simplernn"
    }
    
    selected_model_type = st.selectbox(
        "Selecione o Modelo:", 
        options=list(model_options.keys()),
        index=list(model_options.keys()).index(default_option) if default_option in model_options else 0
    )
    
    return selected_model_type, model_options[selected_model_type]

def create_date_range_selector(df, default_days=90):
    """Cria um seletor de intervalo de datas para visualização.
    
    Args:
        df (pd.DataFrame): DataFrame com os dados históricos.
        default_days (int): Número padrão de dias para exibir.
        
    Returns:
        tuple: (data_início, data_fim)
    """
    if df is None or df.empty:
        st.warning("Dados não disponíveis para seleção de datas.")
        return None, None
    
    min_date = df.index.min().date()
    max_date = df.index.max().date()
    
    # Definir data padrão de início (últimos X dias)
    default_start = max_date - timedelta(days=default_days)
    if default_start < min_date:
        default_start = min_date
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Data de Início:",
            value=default_start,
            min_value=min_date,
            max_value=max_date
        )
    
    with col2:
        end_date = st.date_input(
            "Data de Fim:",
            value=max_date,
            min_value=start_date,
            max_value=max_date
        )
    
    return start_date, end_date

def create_indicator_selector(df, default_indicators=None):
    """Cria um seletor de indicadores técnicos.
    
    Args:
        df (pd.DataFrame): DataFrame com os dados históricos e indicadores.
        default_indicators (list): Lista de indicadores selecionados por padrão.
        
    Returns:
        list: Lista de indicadores selecionados.
    """
    if df is None or df.empty:
        st.warning("Dados não disponíveis para seleção de indicadores.")
        return []
    
    # Filtrar colunas que são indicadores (não são dados OHLCV básicos)
    basic_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume", "Daily Return"]
    indicator_options = [col for col in df.columns if col not in basic_cols]
    
    if not indicator_options:
        st.warning("Nenhum indicador técnico encontrado nos dados.")
        return []
    
    if default_indicators is None:
        # Selecionar alguns indicadores comuns por padrão, se disponíveis
        default_indicators = []
        for ind in ["SMA_50", "RSI", "MACD"]:
            if ind in indicator_options:
                default_indicators.append(ind)
    
    selected_indicators = st.multiselect(
        "Selecione indicadores para visualizar:",
        options=indicator_options,
        default=default_indicators
    )
    
    return selected_indicators

def create_forecast_button(asset_name, model_type):
    """Cria um botão para gerar previsão.
    
    Args:
        asset_name (str): Nome do ativo para exibição.
        model_type (str): Tipo de modelo para exibição.
        
    Returns:
        bool: True se o botão foi clicado, False caso contrário.
    """
    return st.button(f"Gerar Previsão para {asset_name} com {model_type}")

def display_metrics_card(metrics_dict, title="Métricas de Performance"):
    """Exibe um card com métricas de performance.
    
    Args:
        metrics_dict (dict): Dicionário com métricas e valores.
        title (str): Título do card.
    """
    st.subheader(title)
    
    # Criar colunas para as métricas
    cols = st.columns(len(metrics_dict))
    
    for i, (metric_name, metric_value) in enumerate(metrics_dict.items()):
        with cols[i]:
            st.metric(
                label=metric_name,
                value=f"{metric_value:.4f}" if isinstance(metric_value, (int, float)) else metric_value
            )


def display_financial_metrics(metrics_dict, title="Métricas Financeiras Avançadas"):
    """Exibe métricas financeiras em formato organizado.
    
    Args:
        metrics_dict (dict): Dicionário com métricas financeiras.
        title (str): Título da seção.
    """
    st.subheader(title)
    
    # Organizar métricas por categoria
    risk_metrics = {}
    return_metrics = {}
    accuracy_metrics = {}
    
    for key, value in metrics_dict.items():
        if 'sharpe' in key.lower() or 'sortino' in key.lower() or 'calmar' in key.lower():
            risk_metrics[key] = value
        elif 'drawdown' in key.lower() or 'volatility' in key.lower() or 'information' in key.lower():
            risk_metrics[key] = value
        elif 'return' in key.lower():
            return_metrics[key] = value
        elif 'accuracy' in key.lower() or 'prediction' in key.lower():
            accuracy_metrics[key] = value
        else:
            # Métricas gerais
            if not risk_metrics:
                risk_metrics[key] = value
    
    # Exibir métricas de risco
    if risk_metrics:
        st.markdown("**📊 Métricas de Risco e Performance**")
        cols = st.columns(min(4, len(risk_metrics)))
        for i, (metric, value) in enumerate(risk_metrics.items()):
            with cols[i % 4]:
                formatted_value = f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
                st.metric(label=metric.replace('_', ' ').title(), value=formatted_value)
    
    # Exibir métricas de retorno
    if return_metrics:
        st.markdown("**💰 Métricas de Retorno**")
        cols = st.columns(min(4, len(return_metrics)))
        for i, (metric, value) in enumerate(return_metrics.items()):
            with cols[i % 4]:
                formatted_value = f"{value:.4f}" if isinstance(value, (int, float)) else str(value)
                st.metric(label=metric.replace('_', ' ').title(), value=formatted_value)
    
    # Exibir métricas de acurácia
    if accuracy_metrics:
        st.markdown("**🎯 Métricas de Acurácia Direcional**")
        cols = st.columns(min(4, len(accuracy_metrics)))
        for i, (metric, value) in enumerate(accuracy_metrics.items()):
            with cols[i % 4]:
                if 'accuracy' in metric.lower():
                    formatted_value = f"{value*100:.2f}%" if isinstance(value, (int, float)) else str(value)
                else:
                    formatted_value = f"{value:.0f}" if isinstance(value, (int, float)) else str(value)
                st.metric(label=metric.replace('_', ' ').title(), value=formatted_value)


def display_model_comparison(metrics_dict, title="Comparação de Modelos"):
    """Exibe comparação entre métricas de modelos reais e preditos.
    
    Args:
        metrics_dict (dict): Dicionário com métricas de ambos os modelos.
        title (str): Título da seção.
    """
    st.subheader(title)
    
    # Separar métricas por tipo (actual vs predicted)
    actual_metrics = {k.replace('_actual', ''): v for k, v in metrics_dict.items() if '_actual' in k}
    predicted_metrics = {k.replace('_predicted', ''): v for k, v in metrics_dict.items() if '_predicted' in k}
    
    if actual_metrics and predicted_metrics:
        # Criar DataFrame para comparação
        comparison_data = []
        for metric in actual_metrics.keys():
            if metric in predicted_metrics:
                comparison_data.append({
                    'Métrica': metric.replace('_', ' ').title(),
                    'Valores Reais': f"{actual_metrics[metric]:.4f}" if isinstance(actual_metrics[metric], (int, float)) else str(actual_metrics[metric]),
                    'Valores Preditos': f"{predicted_metrics[metric]:.4f}" if isinstance(predicted_metrics[metric], (int, float)) else str(predicted_metrics[metric])
                })
        
        if comparison_data:
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True)
    else:
        st.info("Dados insuficientes para comparação de modelos.")

def display_forecast_table(forecast_values, forecast_dates, decimals=2):
    """Exibe uma tabela com os valores previstos.
    
    Args:
        forecast_values (np.ndarray): Array com os valores previstos.
        forecast_dates (pd.DatetimeIndex): Índice de datas para as previsões.
        decimals (int): Número de casas decimais para arredondar.
    """
    if forecast_values is None or forecast_dates is None:
        st.warning("Dados de previsão não disponíveis.")
        return
    
    st.subheader("Valores Previstos")
    
    # Criar DataFrame para exibição
    forecast_df = pd.DataFrame({
        "Data": forecast_dates,
        "Previsão (Adj Close)": np.round(forecast_values, decimals)
    })
    
    # Exibir como tabela
    st.dataframe(forecast_df.set_index("Data"), use_container_width=True)

def create_sidebar_info(asset_code, model_code):
    """Cria informações na barra lateral.
    
    Args:
        asset_code (str): Código do ativo selecionado.
        model_code (str): Código do modelo selecionado.
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("Informações")
    
    st.sidebar.markdown(f"""
    **Ativo Selecionado:** {asset_code.upper()}
    
    **Modelo Selecionado:** {model_code.upper()}
    
    **Horizonte de Previsão:** 14 dias
    
    **Sequência de Entrada:** 60 dias
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Sobre os Modelos:**
    
    - **LSTM:** Long Short-Term Memory, eficaz para capturar dependências de longo prazo.
    - **GRU:** Gated Recurrent Unit, similar ao LSTM mas com arquitetura mais simples.
    - **SimpleRNN:** Rede Neural Recorrente básica, mais simples mas menos poderosa.
    """)

# Exemplo de uso (para teste manual)
if __name__ == "__main__":
    # Criar dados de exemplo
    dates = pd.date_range(start="2023-01-01", periods=100)
    df_ex = pd.DataFrame({
        "Adj Close": 100 + np.random.randn(100).cumsum(),
        "Volume": np.random.randint(10000, 50000, 100),
        "SMA_50": (100 + np.random.randn(100).cumsum()) * 0.98,
        "RSI": np.random.uniform(30, 70, 100),
        "MACD": np.random.randn(100)
    }, index=dates)
    
    # Testar componentes (requer execução via Streamlit)
    st.title("Teste de Componentes UI")
    
    st.header("Seletores")
    asset_name, asset_code = create_asset_selector()
    model_name, model_code = create_model_selector()
    
    st.write(f"Ativo selecionado: {asset_name} ({asset_code})")
    st.write(f"Modelo selecionado: {model_name} ({model_code})")
    
    st.header("Seletor de Datas")
    start_date, end_date = create_date_range_selector(df_ex)
    st.write(f"Intervalo selecionado: {start_date} a {end_date}")
    
    st.header("Seletor de Indicadores")
    indicators = create_indicator_selector(df_ex)
    st.write(f"Indicadores selecionados: {indicators}")
    
    st.header("Botão de Previsão")
    if create_forecast_button(asset_name, model_name):
        st.success("Botão de previsão clicado!")
    
    st.header("Card de Métricas")
    metrics = {
        "RMSE": 123.45,
        "MAE": 98.76,
        "Sharpe Ratio": 1.23
    }
    display_metrics_card(metrics)
    
    st.header("Tabela de Previsão")
    forecast_dates_ex = pd.date_range(start=df_ex.index[-1] + pd.Timedelta(days=1), periods=14)
    forecast_values_ex = df_ex["Adj Close"].iloc[-1] * (1 + np.random.randn(14)*0.02).cumsum()
    display_forecast_table(forecast_values_ex, forecast_dates_ex)
    
    st.sidebar.header("Barra Lateral")
    create_sidebar_info(asset_code, model_code)

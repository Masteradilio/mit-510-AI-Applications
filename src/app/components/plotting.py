# -*- coding: utf-8 -*-
"""Módulo para plotagem para o aplicativo Streamlit usando Plotly."""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def plot_historical_data(df, asset_name):
    """Plota o preço histórico (Adj Close) e o volume."""
    if df is None or df.empty:
        return go.Figure().update_layout(title="Dados históricos não disponíveis")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                      subplot_titles=(f"{asset_name} - Preço de Fechamento Ajustado", f"{asset_name} - Volume"),
                      vertical_spacing=0.1)

    # Plot Preço
    fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"], mode="lines", name="Adj Close", line=dict(color="#1f77b4")), # Azul
                  row=1, col=1)

    # Plot Volume
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color="#ff7f0e"), # Laranja
                  row=2, col=1)

    fig.update_layout(
        height=500,
        showlegend=False,
        xaxis_rangeslider_visible=False, # Desabilitar rangeslider na subtrama de volume
        margin=dict(l=20, r=20, t=50, b=20) # Ajustar margens
    )
    fig.update_xaxes(rangeslider_visible=True, row=1, col=1) # Habilitar apenas no preço
    return fig

def plot_technical_indicators(df, indicators):
    """Plota os indicadores técnicos selecionados."""
    if df is None or df.empty or not indicators:
        return go.Figure().update_layout(title="Selecione indicadores para visualizar")

    fig = go.Figure()
    colors = ["#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"] # Cores diferentes

    for i, indicator in enumerate(indicators):
        if indicator in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[indicator],
                mode="lines",
                name=indicator,
                line=dict(color=colors[i % len(colors)])
            ))
        else:
            print(f"Aviso: Indicador \"{indicator}\" não encontrado no DataFrame.")

    fig.update_layout(
        title="Indicadores Técnicos",
        xaxis_title="Data",
        yaxis_title="Valor",
        height=400,
        legend_title="Indicadores",
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def plot_forecast_vs_actual(df_history, forecast_values, forecast_dates, asset_name, model_type, history_days=90):
    """Plota o histórico recente e a previsão futura."""
    if df_history is None or df_history.empty or forecast_values is None or forecast_dates is None:
        return go.Figure().update_layout(title="Dados insuficientes para plotar previsão")

    fig = go.Figure()

    # Histórico (últimos N dias para contexto)
    historical_data = df_history["Adj Close"].iloc[-history_days:]
    fig.add_trace(go.Scatter(
        x=historical_data.index,
        y=historical_data,
        mode="lines",
        name="Histórico (Adj Close)",
        line=dict(color="#1f77b4") # Azul
    ))

    # Previsão
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_values,
        mode="lines",
        name=f"Previsão {model_type} ({len(forecast_values)} dias)",
        line=dict(color="#d62728", dash="dash") # Vermelho tracejado
    ))

    fig.update_layout(
        title=f"Previsão de Preço para {asset_name.upper()} com {model_type}",
        xaxis_title="Data",
        yaxis_title="Preço Ajustado (USD)",
        legend_title="Legenda",
        height=500,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

def plot_strategy_performance(returns_dict, title):
    """Plota os retornos acumulados de diferentes estratégias."""
    fig = go.Figure()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"] # Azul, Laranja, Verde, Vermelho
    i = 0
    for name, returns in returns_dict.items():
        if returns is not None and not returns.empty:
            cumulative_returns = (1 + returns).cumprod() - 1
            fig.add_trace(go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns,
                mode="lines",
                name=name,
                line=dict(color=colors[i % len(colors)])
            ))
            i += 1
        else:
            print(f"Aviso: Retornos para \"{name}\" estão vazios ou nulos.")

    fig.update_layout(
        title=title,
        xaxis_title="Data",
        yaxis_title="Retorno Acumulado",
        legend_title="Estratégia",
        height=500,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig

# Exemplo de uso (para teste manual)
if __name__ == "__main__":
    # Criar dados de exemplo
    dates_hist = pd.date_range(start="2023-01-01", periods=100)
    df_ex = pd.DataFrame({
        "Adj Close": 100 + np.random.randn(100).cumsum(),
        "Volume": np.random.randint(10000, 50000, 100),
        "SMA_50": (100 + np.random.randn(100).cumsum()) * 0.98,
        "RSI": np.random.uniform(30, 70, 100)
    }, index=dates_hist)

    dates_fc = pd.date_range(start=df_ex.index[-1] + pd.Timedelta(days=1), periods=14)
    forecast_ex = df_ex["Adj Close"].iloc[-1] * (1 + np.random.randn(14)*0.02).cumsum()

    returns_ex = {
        "Long-Only (Forecast)": pd.Series(np.random.randn(100)*0.01, index=dates_hist),
        "Buy-and-Hold": pd.Series(np.random.randn(100)*0.008, index=dates_hist)
    }

    # Testar plots (requer ambiente gráfico ou salvar como HTML)
    fig1 = plot_historical_data(df_ex, "Exemplo")
    # fig1.show()
    fig2 = plot_technical_indicators(df_ex, ["SMA_50", "RSI"])
    # fig2.show()
    fig3 = plot_forecast_vs_actual(df_ex, forecast_ex, dates_fc, "Exemplo", "LSTM")
    # fig3.show()
    fig4 = plot_strategy_performance(returns_ex, "Performance Exemplo")
    # fig4.show()

    print("Funções de plotagem executadas (verificar visualmente ou salvar HTML).")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
import os

# Adicionar o diretório src ao path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

# try:
from modeling.strategy_simulation import (
    simulate_long_only_strategy,
    generate_signals
)
STRATEGY_MODULE_AVAILABLE = True
# except ImportError:
#     STRATEGY_MODULE_AVAILABLE = False

st.title("📈 Simulação de Estratégias de Trading")
st.markdown("---")

# Sidebar para configurações
st.sidebar.header("⚙️ Configurações da Estratégia")

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

# Seleção da estratégia
strategy_options = {
    "Buy-and-Hold": "buy_hold",
    "Forecast-based Signals (Long-Only)": "forecast_long",
    "Análise de Risco (Sharpe Ratio)": "risk_analysis",
    "Comparação de Performance": "performance_comparison"
}

selected_strategy_name = st.sidebar.selectbox(
    "Selecione a Estratégia:",
    options=list(strategy_options.keys()),
    index=0
)
selected_strategy_code = strategy_options[selected_strategy_name]

# Parâmetros da estratégia
st.sidebar.subheader("Parâmetros")
initial_capital = st.sidebar.number_input(
    "Capital Inicial (USD):",
    min_value=1000,
    max_value=1000000,
    value=10000,
    step=1000
)

threshold = st.sidebar.slider(
    "Limiar de Sinal (%):",
    min_value=0.5,
    max_value=5.0,
    value=1.0,
    step=0.1
) / 100

@st.cache_data
def load_asset_data(asset_code):
    """Carrega os dados do ativo selecionado."""
    try:
        # Corrigir o caminho para ser absoluto
        file_path = os.path.join(PROJECT_ROOT, "data", "processed", f"{asset_code}_processed.csv")
        df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
        
        # Garantir que temos as colunas básicas
        if 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
        
        return df
    except FileNotFoundError:
        st.error(f"Arquivo de dados não encontrado para {asset_code.upper()} em {file_path}")
        st.info(f"Verifique se o arquivo existe em: {os.path.abspath(file_path)}")
        return None

def simulate_strategy_performance(df, strategy_type, threshold, initial_capital):
    """Simula a performance da estratégia selecionada conforme especificações do projeto."""
    if df is None:
        return None, None
    
    prices = df['Close']
    daily_returns = prices.pct_change().dropna()
    
    if strategy_type == "buy_hold":
        # Estratégia Buy-and-Hold: comprar e manter
        returns = daily_returns
        cumulative_returns = (1 + returns).cumprod()
        
    elif strategy_type == "forecast_long":
        # Estratégia Long-Only baseada em previsões (forecast-based signals)
        # Para demonstração, simulamos previsões baseadas em médias móveis
        sma_short = prices.rolling(window=10).mean()
        sma_long = prices.rolling(window=30).mean()
        
        # Gerar sinais: compra quando SMA curta > SMA longa (tendência de alta)
        signals = (sma_short > sma_long).astype(int)
        
        # Aplicar estratégia long-only
        strategy_returns = daily_returns * signals.shift(1)
        returns = strategy_returns.fillna(0)
        cumulative_returns = (1 + returns).cumprod()
        
    elif strategy_type == "risk_analysis":
        # Análise de Risco com Sharpe Ratio
        # Implementar estratégia com gestão de risco
        volatility_window = 30
        rolling_vol = daily_returns.rolling(window=volatility_window).std()
        
        # Ajustar posição baseada na volatilidade (menor posição em alta volatilidade)
        position_size = 1 / (1 + rolling_vol * 10)  # Reduz posição quando volatilidade alta
        returns = daily_returns * position_size.shift(1)
        returns = returns.fillna(0)
        cumulative_returns = (1 + returns).cumprod()
        
    else:  # performance_comparison
        # Comparação de Performance: múltiplas estratégias
        # Buy-and-hold
        bh_returns = daily_returns
        bh_cumulative = (1 + bh_returns).cumprod()
        
        # Estratégia com médias móveis
        sma_short = prices.rolling(window=10).mean()
        sma_long = prices.rolling(window=30).mean()
        ma_signals = (sma_short > sma_long).astype(int)
        ma_returns = daily_returns * ma_signals.shift(1)
        ma_cumulative = (1 + ma_returns.fillna(0)).cumprod()
        
        # Retornar comparação
        returns = pd.DataFrame({
            'Buy-and-Hold': bh_returns,
            'Média Móvel': ma_returns.fillna(0)
        })
        cumulative_returns = pd.DataFrame({
            'Buy-and-Hold': bh_cumulative,
            'Média Móvel': ma_cumulative
        })
        
    # Calcular métricas de performance
    if isinstance(returns, pd.DataFrame):
        # Para comparação de múltiplas estratégias
        metrics = {}
        for col in returns.columns:
            strategy_returns = returns[col]
            total_return = cumulative_returns[col].iloc[-1] - 1
            annual_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
            volatility = strategy_returns.std() * np.sqrt(252)
            sharpe_ratio = annual_return / volatility if volatility > 0 else 0
            max_drawdown = (cumulative_returns[col] / cumulative_returns[col].expanding().max() - 1).min()
            
            metrics[col] = {
                "Retorno Total": f"{total_return:.2%}",
                "Retorno Anualizado": f"{annual_return:.2%}",
                "Volatilidade": f"{volatility:.2%}",
                "Sharpe Ratio": f"{sharpe_ratio:.2f}",
                "Max Drawdown": f"{max_drawdown:.2%}",
                "Capital Final": f"${initial_capital * (1 + total_return):,.2f}"
            }
    else:
        # Para estratégia única
        total_return = cumulative_returns.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        max_drawdown = (cumulative_returns / cumulative_returns.expanding().max() - 1).min()
        
        metrics = {
            "Retorno Total": f"{total_return:.2%}",
            "Retorno Anualizado": f"{annual_return:.2%}",
            "Volatilidade Anualizada": f"{volatility:.2%}",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}",
            "Máximo Drawdown": f"{max_drawdown:.2%}",
            "Capital Final": f"${initial_capital * (1 + total_return):,.2f}"
        }
    
    return cumulative_returns, metrics

# Interface principal
st.header(f"📈 Análise de Estratégia: {selected_strategy_name}")
st.subheader(f"Ativo: {selected_asset_name}")

# Mostrar aviso se módulo não estiver disponível
if not STRATEGY_MODULE_AVAILABLE:
    st.warning("⚠️ Módulo de simulação de estratégias não encontrado. Usando simulação básica.")

# Carregar dados automaticamente
with st.spinner(f"Carregando dados de {selected_asset_name}..."):
    df = load_asset_data(selected_asset_code)

if df is not None:
    # Simular estratégia automaticamente
    with st.spinner("Simulando estratégia..."):
        cumulative_returns, metrics = simulate_strategy_performance(
            df, selected_strategy_code, threshold, initial_capital
        )
    
    if cumulative_returns is not None and metrics is not None:
        # Exibir métricas principais
        st.subheader("📊 Métricas de Performance")
        
        if isinstance(metrics, dict) and not any(isinstance(v, dict) for v in metrics.values()):
            # Estratégia única
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Retorno Total", metrics["Retorno Total"])
                st.metric("Sharpe Ratio", metrics["Sharpe Ratio"])
            
            with col2:
                st.metric("Retorno Anualizado", metrics["Retorno Anualizado"])
                st.metric("Máximo Drawdown", metrics["Máximo Drawdown"])
            
            with col3:
                st.metric("Volatilidade", metrics["Volatilidade Anualizada"])
                st.metric("Capital Final", metrics["Capital Final"])
        else:
            # Comparação de múltiplas estratégias
            for strategy_name, strategy_metrics in metrics.items():
                st.subheader(f"📈 {strategy_name}")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Retorno Total", strategy_metrics["Retorno Total"])
                    st.metric("Sharpe Ratio", strategy_metrics["Sharpe Ratio"])
                
                with col2:
                    st.metric("Retorno Anualizado", strategy_metrics["Retorno Anualizado"])
                    st.metric("Max Drawdown", strategy_metrics["Max Drawdown"])
                
                with col3:
                    st.metric("Volatilidade", strategy_metrics["Volatilidade"])
                    st.metric("Capital Final", strategy_metrics["Capital Final"])
        
        # Gráfico de performance
        st.subheader("📈 Evolução do Capital")
        
        fig = go.Figure()
        
        if isinstance(cumulative_returns, pd.DataFrame):
            # Múltiplas estratégias
            colors = ['blue', 'green', 'red', 'orange']
            for i, col in enumerate(cumulative_returns.columns):
                strategy_value = initial_capital * cumulative_returns[col]
                fig.add_trace(go.Scatter(
                    x=cumulative_returns.index,
                    y=strategy_value,
                    mode='lines',
                    name=col,
                    line=dict(color=colors[i % len(colors)], width=2)
                ))
        else:
            # Estratégia única
            strategy_value = initial_capital * cumulative_returns
            fig.add_trace(go.Scatter(
                x=cumulative_returns.index,
                y=strategy_value,
                mode='lines',
                name=f'Estratégia: {selected_strategy_name}',
                line=dict(color='blue', width=2)
            ))
            
            # Buy and Hold para comparação (apenas para estratégias únicas)
            if selected_strategy_code != "buy_hold":
                buy_hold_returns = df['Close'] / df['Close'].iloc[0]
                buy_hold_value = initial_capital * buy_hold_returns
                fig.add_trace(go.Scatter(
                    x=buy_hold_returns.index,
                    y=buy_hold_value,
                    mode='lines',
                    name='Buy & Hold (Referência)',
                    line=dict(color='gray', width=2, dash='dash')
                ))
        
        fig.update_layout(
            title=f"Performance das Estratégias - {selected_asset_name}",
            xaxis_title="Data",
            yaxis_title="Valor do Portfólio (USD)",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Tabela detalhada de métricas
        st.subheader("📋 Métricas Detalhadas")
        
        if isinstance(metrics, dict) and not any(isinstance(v, dict) for v in metrics.values()):
            # Estratégia única
            metrics_df = pd.DataFrame({
                "Métrica": list(metrics.keys()),
                "Valor": list(metrics.values())
            })
            st.dataframe(metrics_df, use_container_width=True)
        else:
            # Múltiplas estratégias
            comparison_data = []
            for strategy_name, strategy_metrics in metrics.items():
                for metric_name, metric_value in strategy_metrics.items():
                    comparison_data.append({
                        "Estratégia": strategy_name,
                        "Métrica": metric_name,
                        "Valor": metric_value
                    })
            
            comparison_df = pd.DataFrame(comparison_data)
            pivot_df = comparison_df.pivot(index="Métrica", columns="Estratégia", values="Valor")
            st.dataframe(pivot_df, use_container_width=True)
        
        # Análise de drawdown
        st.subheader("📉 Análise de Drawdown")
        
        fig_dd = go.Figure()
        
        if isinstance(cumulative_returns, pd.DataFrame):
            # Múltiplas estratégias
            colors = ['red', 'orange', 'purple', 'brown']
            for i, col in enumerate(cumulative_returns.columns):
                drawdown = (cumulative_returns[col] / cumulative_returns[col].expanding().max() - 1) * 100
                fig_dd.add_trace(go.Scatter(
                    x=drawdown.index,
                    y=drawdown,
                    mode='lines',
                    name=f'Drawdown {col}',
                    line=dict(color=colors[i % len(colors)])
                ))
        else:
            # Estratégia única
            drawdown = (cumulative_returns / cumulative_returns.expanding().max() - 1) * 100
            fig_dd.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown,
                mode='lines',
                fill='tonexty',
                name='Drawdown (%)',
                line=dict(color='red')
            ))
        
        fig_dd.update_layout(
            title="Evolução do Drawdown",
            xaxis_title="Data",
            yaxis_title="Drawdown (%)",
            height=400
        )
        
        st.plotly_chart(fig_dd, use_container_width=True)
        
    else:
        st.error("Erro ao simular a estratégia. Verifique os dados e parâmetros.")
else:
    st.error("❌ Não foi possível carregar os dados do ativo selecionado.")
    st.info("💡 **Dica:** Verifique se os arquivos de dados processados existem.")

# Informações sobre estratégias
with st.expander("ℹ️ Sobre as Estratégias"):
    st.markdown("""
    ### Estratégias Disponíveis:
    
    **1. Long Only (Baseada em Previsões):**
    - Utiliza previsões de modelos de ML para gerar sinais de compra/venda
    - Mantém posições apenas compradas (long)
    - Ideal para mercados em tendência de alta
    
    **2. Estratégia Técnica:**
    - Baseada em indicadores técnicos (RSI, MACD, Bollinger Bands)
    - Combina múltiplos sinais para decisões de trading
    - Adequada para mercados com padrões técnicos claros
    
    **3. Estratégia de Momentum:**
    - Segue a tendência de preços de curto prazo
    - Compra em alta e vende em baixa
    - Funciona bem em mercados com tendências fortes
    
    **4. Reversão à Média:**
    - Assume que preços retornam à média histórica
    - Compra quando preços estão baixos, vende quando altos
    - Eficaz em mercados laterais ou com reversões frequentes
    """)

st.markdown("---")
st.markdown("**MIT-510 - Módulo de Estratégias de Trading**")
st.markdown("Análise quantitativa de estratégias baseadas em Machine Learning")
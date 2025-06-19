import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
import sys

from statsmodels.tsa.seasonal import seasonal_decompose
from arch import arch_model
import seaborn as sns
from statsmodels.tsa.stattools import coint, grangercausalitytests

# Título principal
st.title("📊 Análise Exploratória de Dados (EDA)")
st.markdown("--- ")

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
    index=0,
    key='eda_asset_selectbox' # Chave única para este widget
)
selected_asset_code = asset_options[selected_asset_name]

@st.cache_data
def load_processed_data_eda(asset_name):
    """Carrega os dados processados do ativo selecionado para EDA."""
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_script_path, "..", "..", ".."))
    file_path = os.path.join(project_root, "data", "processed", f"{asset_name}_processed.csv")
    try:
        df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
        if 'Adj Close' in df.columns:
            df['Close'] = df['Adj Close']
        
        # Adicionar indicadores para EDA se não existirem
        if "SMA_50" not in df.columns:
            df["SMA_50"] = df["Close"].rolling(window=50, min_periods=1).mean()
        if "SMA_200" not in df.columns: # Adicionando SMA_200 para mais análise
            df["SMA_200"] = df["Close"].rolling(window=200, min_periods=1).mean()
        if "RSI" not in df.columns:
            delta = df['Close'].diff(1)
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            df['RSI'] = df['RSI'].fillna(50)
        
        # Calcular Volatilidade (desvio padrão dos retornos diários)
        if 'Daily_Return' not in df.columns:
            df['Daily_Return'] = df['Close'].pct_change()
        if 'Volatility_30D' not in df.columns:
            df['Volatility_30D'] = df['Daily_Return'].rolling(window=30).std() * np.sqrt(252) # Anualizada
            df['Volatility_30D'] = df['Volatility_30D'].fillna(method='bfill')

        df = df.fillna(method='ffill').fillna(method='bfill')
        return df
    except FileNotFoundError:
        st.error(f"Arquivo de dados processados não encontrado para {asset_name.upper()} em {file_path}.")
        return None
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar ou processar os dados para {asset_name.upper()}: {e}")
        return None

# Carregar dados
df_eda = load_processed_data_eda(selected_asset_code)

if df_eda is not None:
    st.subheader(f"Análise Exploratória para {selected_asset_name}")

    # 1. Visualização da Série Temporal do Preço de Fechamento
    st.markdown("### 1. Série Temporal do Preço de Fechamento")
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=df_eda.index, y=df_eda['Close'], mode='lines', name='Preço de Fechamento'))
    fig_price.add_trace(go.Scatter(x=df_eda.index, y=df_eda['SMA_50'], mode='lines', name='SMA 50 dias', line=dict(dash='dash')))
    fig_price.add_trace(go.Scatter(x=df_eda.index, y=df_eda['SMA_200'], mode='lines', name='SMA 200 dias', line=dict(dash='dot')))
    fig_price.update_layout(title=f'Preço de Fechamento e Médias Móveis - {selected_asset_name}',
                            xaxis_title='Data',
                            yaxis_title='Preço de Fechamento (USD)',
                            legend_title='Legenda')
    st.plotly_chart(fig_price, use_container_width=True)
    st.markdown("Este gráfico mostra a evolução do preço de fechamento do ativo ao longo do tempo, juntamente com as médias móveis de 50 e 200 dias, que ajudam a identificar tendências de curto e longo prazo, respectivamente.")

    # 2. Volume de Negociação
    st.markdown("### 2. Volume de Negociação")
    fig_volume = px.bar(df_eda, x=df_eda.index, y='Volume', title=f'Volume de Negociação - {selected_asset_name}')
    fig_volume.update_layout(xaxis_title='Data', yaxis_title='Volume Negociado')
    st.plotly_chart(fig_volume, use_container_width=True)
    st.markdown("O volume de negociação indica a quantidade de um ativo que foi transacionada em um determinado período. Picos de volume podem sinalizar eventos importantes ou mudanças na percepção do mercado.")

    # 3. Distribuição dos Retornos Diários
    st.markdown("### 3. Distribuição dos Retornos Diários")
    if 'Daily_Return' in df_eda.columns:
        fig_returns_dist = px.histogram(df_eda, x='Daily_Return', nbins=100, title=f'Distribuição dos Retornos Diários - {selected_asset_name}')
        fig_returns_dist.update_layout(xaxis_title='Retorno Diário', yaxis_title='Frequência')
        st.plotly_chart(fig_returns_dist, use_container_width=True)
        st.markdown("Este histograma mostra a frequência dos diferentes níveis de retornos diários. Ajuda a entender a volatilidade e o risco associado ao ativo. Uma distribuição mais concentrada em torno de zero indica menor volatilidade.")
    else:
        st.warning("Coluna 'Daily_Return' não encontrada para gerar o histograma de retornos.")

    # 4. Volatilidade Histórica (Rolling Standard Deviation)
    st.markdown("### 4. Volatilidade Histórica (30 dias)")
    if 'Volatility_30D' in df_eda.columns:
        fig_volatility = go.Figure()
        fig_volatility.add_trace(go.Scatter(x=df_eda.index, y=df_eda['Volatility_30D'], mode='lines', name='Volatilidade (30D)'))
        fig_volatility.update_layout(title=f'Volatilidade Histórica (Rolling 30 dias) - {selected_asset_name}',
                                xaxis_title='Data',
                                yaxis_title='Volatilidade Anualizada')
        st.plotly_chart(fig_volatility, use_container_width=True)
        st.markdown("A volatilidade histórica, calculada como o desvio padrão dos retornos diários em uma janela móvel (aqui, 30 dias) e anualizada, mede o grau de variação do preço de um ativo. Períodos de alta volatilidade indicam maior incerteza ou risco.")
    else:
        st.warning("Coluna 'Volatility_30D' não encontrada para gerar o gráfico de volatilidade.")

    # 5. Indicador de Força Relativa (RSI)
    st.markdown("### 5. Indicador de Força Relativa (RSI)")
    if 'RSI' in df_eda.columns:
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df_eda.index, y=df_eda['RSI'], mode='lines', name='RSI'))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Sobrevendido (70)", annotation_position="bottom right")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Sobrecomprado (30)", annotation_position="bottom right")
        fig_rsi.update_layout(title=f'Indicador de Força Relativa (RSI) - {selected_asset_name}',
                            xaxis_title='Data',
                            yaxis_title='RSI (0-100)')
        st.plotly_chart(fig_rsi, use_container_width=True)
        st.markdown("O RSI é um oscilador de momento que mede a velocidade e a mudança dos movimentos de preços. Valores acima de 70 geralmente indicam que um ativo está sobrecomprado (pode estar prestes a cair), enquanto valores abaixo de 30 indicam que está sobrevendido (pode estar prestes a subir).")
    else:
        st.warning("Coluna 'RSI' não encontrada para gerar o gráfico de RSI.")

    # 6. Estatísticas Descritivas
    st.markdown("### 6. Estatísticas Descritivas dos Dados Processados")
    st.dataframe(df_eda.describe(), use_container_width=True)
    st.markdown("A tabela acima apresenta um resumo estatístico das principais colunas dos dados processados, incluindo contagem, média, desvio padrão, valores mínimos e máximos, e os quartis. Isso fornece uma visão geral da distribuição e das características centrais dos dados.")

    # 7. Análise de Sazonalidade e Tendência
    st.markdown("### 7. Análise de Sazonalidade e Tendência")
    if 'Close' in df_eda.columns and not df_eda['Close'].empty:
        try:
            # Usar um período menor se os dados forem insuficientes para 365 dias
            period_seasonality = min(len(df_eda['Close']) // 2, 365 if selected_asset_code == 'aapl' else 30) 
            if period_seasonality < 2:
                st.warning("Não há dados suficientes para realizar a decomposição sazonal.")
            else:
                decomposition = seasonal_decompose(df_eda['Close'], model='additive', period=period_seasonality)
                
                fig_seasonal = make_subplots(rows=4, cols=1, 
                                           shared_xaxes=True, 
                                           subplot_titles=('Observado', 'Tendência', 'Sazonalidade', 'Residual'))
                
                fig_seasonal.add_trace(go.Scatter(x=df_eda.index, y=decomposition.observed, name='Observado'), row=1, col=1)
                fig_seasonal.add_trace(go.Scatter(x=df_eda.index, y=decomposition.trend, name='Tendência'), row=2, col=1)
                fig_seasonal.add_trace(go.Scatter(x=df_eda.index, y=decomposition.seasonal, name='Sazonalidade'), row=3, col=1)
                fig_seasonal.add_trace(go.Scatter(x=df_eda.index, y=decomposition.resid, name='Residual'), row=4, col=1)
                
                fig_seasonal.update_layout(height=800, title_text=f'Decomposição Sazonal - {selected_asset_name}')
                st.plotly_chart(fig_seasonal, use_container_width=True)
                st.markdown("A decomposição sazonal divide a série temporal em seus componentes: tendência (direção de longo prazo), sazonalidade (padrões que se repetem em intervalos fixos) e resíduos (ruído aleatório). Isso ajuda a entender os diferentes fatores que influenciam o preço do ativo.")
        except Exception as e:
            st.error(f"Erro ao realizar a decomposição sazonal: {e}")
    else:
        st.warning("Coluna 'Close' não encontrada ou vazia para análise de sazonalidade.")

    # 8. Análise de Volatilidade com GARCH(1,1)
    st.markdown("### 8. Análise de Volatilidade com GARCH(1,1)")
    if 'Daily_Return' in df_eda.columns and not df_eda['Daily_Return'].dropna().empty:
        try:
            returns_for_garch = df_eda['Daily_Return'].dropna() * 100 # GARCH geralmente funciona melhor com retornos percentuais
            if len(returns_for_garch) < 20: # GARCH precisa de um número mínimo de observações
                st.warning("Não há dados suficientes para estimar o modelo GARCH.")
            else:
                model_garch = arch_model(returns_for_garch, vol='Garch', p=1, q=1, rescale=False)
                results_garch = model_garch.fit(disp='off')
                
                st.subheader("Sumário do Modelo GARCH(1,1)")
                st.text(str(results_garch.summary()))
                
                fig_garch_vol = go.Figure()
                fig_garch_vol.add_trace(go.Scatter(x=results_garch.conditional_volatility.index, 
                                                   y=results_garch.conditional_volatility, 
                                                   name='Volatilidade Condicional Estimada (GARCH)'))
                fig_garch_vol.update_layout(title=f'Volatilidade Condicional Estimada pelo GARCH(1,1) - {selected_asset_name}',
                                        xaxis_title='Data',
                                        yaxis_title='Volatilidade Condicional')
                st.plotly_chart(fig_garch_vol, use_container_width=True)
                st.markdown("O modelo GARCH (Generalized Autoregressive Conditional Heteroskedasticity) é usado para modelar e prever a volatilidade dos retornos financeiros. O gráfico acima mostra a volatilidade condicional estimada, que captura os clusters de volatilidade (períodos de alta volatilidade tendem a ser seguidos por alta volatilidade, e vice-versa).")
        except Exception as e:
            st.error(f"Erro ao estimar o modelo GARCH: {e}. Verifique se há dados suficientes e se os retornos não são constantes.")
    else:
        st.warning("Coluna 'Daily_Return' não encontrada ou vazia para análise GARCH.")

    # 9. Análise de Clusters de Volatilidade
    st.markdown("### 9. Análise de Clusters de Volatilidade")
    if 'Daily_Return' in df_eda.columns and 'Volatility_30D' in df_eda.columns:
        fig_vol_clusters = px.scatter(df_eda, x='Daily_Return', y='Volatility_30D', 
                                        title=f'Clusters de Volatilidade: Retorno Diário vs. Volatilidade (30D) - {selected_asset_name}',
                                        labels={'Daily_Return': 'Retorno Diário', 'Volatility_30D': 'Volatilidade (30D)'},
                                        trendline='ols', trendline_color_override='red')
        fig_vol_clusters.update_layout(xaxis_title='Retorno Diário',
                                       yaxis_title='Volatilidade Anualizada (30 dias)')
        st.plotly_chart(fig_vol_clusters, use_container_width=True)
        st.markdown("Este gráfico de dispersão mostra a relação entre os retornos diários e a volatilidade de 30 dias. "
                    "Pode ajudar a identificar visualmente os chamados 'clusters de volatilidade', onde períodos de alta volatilidade "
                    "tendem a coincidir com movimentos de preços mais extremos (positivos ou negativos), e vice-versa. "
                    "A linha de tendência (OLS) pode indicar uma correlação geral entre as duas métricas.")
    else:
        st.warning("Colunas 'Daily_Return' ou 'Volatility_30D' não encontradas para a análise de clusters de volatilidade.")

    # 10. Matriz de Correlação
    st.markdown("### 10. Matriz de Correlação")
    st.markdown("A matriz de correlação mostra a relação linear entre diferentes variáveis. Valores próximos de 1 indicam uma forte correlação positiva, próximos de -1 uma forte correlação negativa, e próximos de 0 pouca ou nenhuma correlação linear.")
    
    # Selecionar colunas relevantes para a correlação
    cols_for_corr = ['Close', 'Volume', 'Daily_Return', 'Volatility_30D', 'SMA_50', 'SMA_200', 'RSI']
    df_corr = df_eda[cols_for_corr].copy()
    df_corr.rename(columns={
        'Close': 'Preço Fechamento',
        'Volume': 'Volume Negociado',
        'Daily_Return': 'Retorno Diário',
        'Volatility_30D': 'Volatilidade (30D)',
        'SMA_50': 'Média Móvel 50D',
        'SMA_200': 'Média Móvel 200D',
        'RSI': 'IFR (RSI)'
    }, inplace=True)

    if not df_corr.empty:
        corr_matrix = df_corr.corr()
        fig_corr = go.Figure(data=go.Heatmap(
                           z=corr_matrix.values,
                           x=corr_matrix.columns,
                           y=corr_matrix.columns,
                           colorscale='RdBu',
                           zmin=-1, zmax=1,
                           text=corr_matrix.round(2).values,
                           texttemplate="%{text}",
                           hoverongaps = False))
        fig_corr.update_layout(title=f'Matriz de Correlação - {selected_asset_name}',
                               height=600)
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("Não foi possível gerar a matriz de correlação devido à falta de dados.")

    # 11. Análise de Cointegração (entre AAPL e BTC)
    st.markdown("### 11. Análise de Cointegração (entre Apple e Bitcoin)")
    st.markdown("A cointegração testa se duas ou mais séries temporais, que são individualmente não estacionárias (têm tendências), possuem uma combinação linear que é estacionária. Se cointegradas, elas têm um relacionamento de equilíbrio de longo prazo.")
    
    # Carregar dados do outro ativo para cointegração
    other_asset_code = 'btc' if selected_asset_code == 'aapl' else 'aapl'
    other_asset_name_display = asset_options["Bitcoin (BTC-USD)"] if selected_asset_code == 'aapl' else asset_options["Apple (AAPL)"]
    df_other_eda = load_processed_data_eda(other_asset_code)

    if df_other_eda is not None and 'Close' in df_eda.columns and 'Close' in df_other_eda.columns:
        # Alinhar os dataframes pela data
        df_merged_coint = pd.merge(df_eda[['Close']].rename(columns={'Close': selected_asset_name}), 
                                   df_other_eda[['Close']].rename(columns={'Close': other_asset_name_display}), 
                                   left_index=True, right_index=True, how='inner')
        
        if len(df_merged_coint) > 20: # Necessário um número mínimo de observações
            try:
                # Remover NaNs que podem surgir do merge ou de dados originais
                df_merged_coint.dropna(inplace=True)
                if len(df_merged_coint) > 20:
                    coint_score, p_value, crit_value = coint(df_merged_coint[selected_asset_name], df_merged_coint[other_asset_name_display])
                    st.write(f"**Resultado do Teste de Cointegração de Engle-Granger entre {selected_asset_name} e {other_asset_name_display}:**")
                    st.write(f"- Estatística do Teste (Score): {coint_score:.4f}")
                    st.write(f"- P-valor: {p_value:.4f}")
                    st.write("Valores Críticos:")
                    st.write(f"  - 1%: {crit_value[0]:.4f}")
                    st.write(f"  - 5%: {crit_value[1]:.4f}")
                    st.write(f"  - 10%: {crit_value[2]:.4f}")
                    if p_value < 0.05:
                        st.success(f"Conclusão: Há evidência de cointegração entre {selected_asset_name} e {other_asset_name_display} ao nível de significância de 5% (p-valor < 0.05). Isso sugere uma relação de equilíbrio de longo prazo.")
                    else:
                        st.warning(f"Conclusão: Não há evidência suficiente de cointegração entre {selected_asset_name} e {other_asset_name_display} ao nível de significância de 5% (p-valor >= 0.05).")
                else:
                    st.warning("Não há dados suficientes após o alinhamento e remoção de NaNs para realizar o teste de cointegração.")
            except Exception as e:
                st.error(f"Erro ao realizar o teste de cointegração: {e}")
        else:
            st.warning("Não há dados suficientes após o alinhamento para realizar o teste de cointegração.")
    else:
        st.warning(f"Não foi possível carregar os dados de {other_asset_name_display} para a análise de cointegração ou faltam dados de fechamento.")

    # 12. Análise de Causalidade de Granger (entre AAPL e BTC)
    st.markdown("### 12. Análise de Causalidade de Granger")
    st.markdown("O teste de causalidade de Granger verifica se os valores passados de uma série temporal X ajudam a prever os valores futuros de uma série temporal Y. Não implica causalidade no sentido filosófico, mas sim uma capacidade de previsão.")

    if df_other_eda is not None and 'Daily_Return' in df_eda.columns and 'Daily_Return' in df_other_eda.columns:
        df_merged_granger = pd.merge(df_eda[['Daily_Return']].rename(columns={'Daily_Return': f'{selected_asset_code}_returns'}), 
                                     df_other_eda[['Daily_Return']].rename(columns={'Daily_Return': f'{other_asset_code}_returns'}), 
                                     left_index=True, right_index=True, how='inner')
        df_merged_granger.dropna(inplace=True)

        if len(df_merged_granger) > 20:
            max_lag = 5 # Definir um lag máximo para o teste
            st.write(f"**Teste de Causalidade de Granger entre os retornos diários de {selected_asset_name} e {other_asset_name_display} (max_lag={max_lag}):**")
            
            try:
                st.write(f"Hipótese 1: {other_asset_name_display} NÃO causa Granger {selected_asset_name}")
                granger_results_1 = grangercausalitytests(df_merged_granger[[f'{selected_asset_code}_returns', f'{other_asset_code}_returns']], maxlag=max_lag, verbose=False)
                # Extrair e apresentar p-valores de forma mais clara
                p_values_1 = [granger_results_1[lag][0]['ssr_ftest'][1] for lag in granger_results_1]
                min_p_value_1 = min(p_values_1) if p_values_1 else 1.0
                st.write(f"Menor p-valor encontrado para H1 (lags 1 a {max_lag}): {min_p_value_1:.4f}")
                if min_p_value_1 < 0.05:
                    st.success(f"Rejeita-se H0: {other_asset_name_display} CAUSA Granger {selected_asset_name} (p < 0.05).")
                else:
                    st.warning(f"Não se rejeita H0: {other_asset_name_display} NÃO causa Granger {selected_asset_name} (p >= 0.05).")

                st.write(f"Hipótese 2: {selected_asset_name} NÃO causa Granger {other_asset_name_display}")
                granger_results_2 = grangercausalitytests(df_merged_granger[[f'{other_asset_code}_returns', f'{selected_asset_code}_returns']], maxlag=max_lag, verbose=False)
                p_values_2 = [granger_results_2[lag][0]['ssr_ftest'][1] for lag in granger_results_2]
                min_p_value_2 = min(p_values_2) if p_values_2 else 1.0
                st.write(f"Menor p-valor encontrado para H2 (lags 1 a {max_lag}): {min_p_value_2:.4f}")
                if min_p_value_2 < 0.05:
                    st.success(f"Rejeita-se H0: {selected_asset_name} CAUSA Granger {other_asset_name_display} (p < 0.05).")
                else:
                    st.warning(f"Não se rejeita H0: {selected_asset_name} NÃO causa Granger {other_asset_name_display} (p >= 0.05).")
                st.caption("Nota: Um p-valor baixo (< 0.05) sugere que os valores passados de uma série ajudam a prever a outra.")
            except Exception as e:
                st.error(f"Erro ao realizar o teste de causalidade de Granger: {e}")
        else:
            st.warning("Não há dados suficientes após o alinhamento para realizar o teste de causalidade de Granger.")
    else:
        st.warning(f"Não foi possível carregar os dados de {other_asset_name_display} ou faltam dados de retornos diários para a análise de causalidade de Granger.")

else:
    st.warning("Não foi possível carregar os dados para a análise exploratória.")

st.sidebar.markdown("--- ")
st.sidebar.info("Esta página realiza uma análise exploratória dos dados históricos do ativo selecionado, mostrando gráficos de preço, volume, retornos, volatilidade, RSI, GARCH e clusters de volatilidade.")
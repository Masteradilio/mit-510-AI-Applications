# -*- coding: utf-8 -*-
"""Módulo para engenharia de features e cálculo de indicadores técnicos."""

import pandas as pd
import numpy as np
import ta
import yfinance as yf
from datetime import datetime, timedelta

def add_technical_indicators(df, include_legacy=True):
    """
    Adiciona indicadores técnicos ao DataFrame conforme documento de qualidade.
    
    Indicadores implementados (conforme documento):
    - Bollinger Bands (BB_Upper, BB_Lower, BB_Position)
    - Stochastic Oscillator (Stoch_K, Stoch_D)
    - Williams %R
    - Commodity Channel Index (CCI)
    - Average Directional Index (ADX)
    - On-Balance Volume (OBV)
    - Average True Range (ATR)
    - Momentum
    - Rate of Change (ROC)
    - TRIX
    
    Args:
        df: DataFrame com colunas OHLCV
        include_legacy: Se True, inclui indicadores legados (SMA, EMA, RSI, MACD)
    
    Returns:
        DataFrame com indicadores técnicos adicionados
    """
    df_result = df.copy()
    
    # Verificar se temos as colunas necessárias
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df_result.columns]
    if missing_cols:
        raise ValueError(f"Colunas faltando: {missing_cols}")
    
    # Calcular retornos diários
    df_result['Returns'] = df_result['Close'].pct_change()
    
    # === INDICADORES TÉCNICOS CONFORME DOCUMENTO DE QUALIDADE ===
    
    # 1. Bollinger Bands (Bandas de Bollinger)
    bollinger = ta.volatility.BollingerBands(df_result['Close'], window=20, window_dev=2)
    df_result['BB_Upper'] = bollinger.bollinger_hband()
    df_result['BB_Lower'] = bollinger.bollinger_lband()
    df_result['BB_Middle'] = bollinger.bollinger_mavg()
    df_result['BB_Width'] = (df_result['BB_Upper'] - df_result['BB_Lower']) / df_result['BB_Middle']
    df_result['BB_Position'] = (df_result['Close'] - df_result['BB_Lower']) / (df_result['BB_Upper'] - df_result['BB_Lower'])
    
    # 2. Stochastic Oscillator (Oscilador Estocástico)
    stoch = ta.momentum.StochasticOscillator(df_result['High'], df_result['Low'], df_result['Close'], window=14, smooth_window=3)
    df_result['Stoch_K'] = stoch.stoch()
    df_result['Stoch_D'] = stoch.stoch_signal()
    
    # 3. Williams %R
    df_result['Williams_R'] = ta.momentum.williams_r(df_result['High'], df_result['Low'], df_result['Close'], lbp=14)
    
    # 4. CCI (Commodity Channel Index)
    df_result['CCI'] = ta.trend.cci(df_result['High'], df_result['Low'], df_result['Close'], window=20)
    
    # 5. ADX (Average Directional Index)
    df_result['ADX'] = ta.trend.adx(df_result['High'], df_result['Low'], df_result['Close'], window=14)
    df_result['ADX_Pos'] = ta.trend.adx_pos(df_result['High'], df_result['Low'], df_result['Close'], window=14)
    df_result['ADX_Neg'] = ta.trend.adx_neg(df_result['High'], df_result['Low'], df_result['Close'], window=14)
    
    # 6. OBV (On-Balance Volume)
    df_result['OBV'] = ta.volume.on_balance_volume(df_result['Close'], df_result['Volume'])
    df_result['OBV_SMA'] = ta.trend.sma_indicator(df_result['OBV'], window=10)
    
    # 7. ATR (Average True Range)
    df_result['ATR'] = ta.volatility.average_true_range(df_result['High'], df_result['Low'], df_result['Close'], window=14)
    df_result['ATR_Ratio'] = df_result['ATR'] / df_result['Close']
    
    # 8. Momentum
    df_result['Momentum_10'] = df_result['Close'] - df_result['Close'].shift(10)
    df_result['Momentum_20'] = df_result['Close'] - df_result['Close'].shift(20)
    
    # 9. ROC (Rate of Change)
    df_result['ROC_10'] = ta.momentum.roc(df_result['Close'], window=10)
    df_result['ROC_20'] = ta.momentum.roc(df_result['Close'], window=20)
    
    # 10. TRIX
    df_result['TRIX'] = ta.trend.trix(df_result['Close'], window=14)
    
    # === INDICADORES LEGADOS (OPCIONAIS) ===
    if include_legacy:
        # Médias Móveis Simples (SMA)
        df_result['SMA_20'] = ta.trend.sma_indicator(df_result['Close'], window=20)
        df_result['SMA_50'] = ta.trend.sma_indicator(df_result['Close'], window=50)
        df_result['SMA_200'] = ta.trend.sma_indicator(df_result['Close'], window=200)
        
        # Médias Móveis Exponenciais (EMA)
        df_result['EMA_12'] = ta.trend.ema_indicator(df_result['Close'], window=12)
        df_result['EMA_26'] = ta.trend.ema_indicator(df_result['Close'], window=26)
        
        # RSI (Relative Strength Index)
        df_result['RSI'] = ta.momentum.rsi(df_result['Close'], window=14)
        
        # MACD (Moving Average Convergence Divergence)
        macd = ta.trend.MACD(df_result['Close'], window_slow=26, window_fast=12, window_sign=9)
        df_result['MACD'] = macd.macd()
        df_result['MACD_Signal'] = macd.macd_signal()
        df_result['MACD_Diff'] = macd.macd_diff()
        
        # Cruzamentos de médias móveis
        df_result['SMA_20_50_Cross'] = (df_result['SMA_20'] > df_result['SMA_50']).astype(int)
        df_result['EMA_12_26_Cross'] = (df_result['EMA_12'] > df_result['EMA_26']).astype(int)
        
        # Indicadores de tendência
        df_result['Price_SMA_20_Ratio'] = df_result['Close'] / df_result['SMA_20']
        df_result['Price_SMA_50_Ratio'] = df_result['Close'] / df_result['SMA_50']
    
    # === FEATURES DERIVADAS ===
    
    # Volatilidade histórica
    df_result['Volatility_10d'] = df_result['Returns'].rolling(window=10).std() * np.sqrt(252)
    df_result['Volatility_30d'] = df_result['Returns'].rolling(window=30).std() * np.sqrt(252)
    
    # Sinais de sobrecompra/sobrevenda
    df_result['Stoch_Oversold'] = (df_result['Stoch_K'] < 20).astype(int)
    df_result['Stoch_Overbought'] = (df_result['Stoch_K'] > 80).astype(int)
    df_result['Williams_Oversold'] = (df_result['Williams_R'] < -80).astype(int)
    df_result['Williams_Overbought'] = (df_result['Williams_R'] > -20).astype(int)
    
    # Força da tendência baseada no ADX
    df_result['ADX_Strong_Trend'] = (df_result['ADX'] > 25).astype(int)
    df_result['ADX_Weak_Trend'] = (df_result['ADX'] < 20).astype(int)
    
    # Divergências de volume
    df_result['Volume_SMA'] = ta.trend.sma_indicator(df_result['Volume'], window=20)
    df_result['Volume_Ratio'] = df_result['Volume'] / df_result['Volume_SMA']
    
    return df_result


def add_exogenous_variables(df, use_comprehensive_collector=True, symbols=['SPY', 'QQQ', 'VIX', 'DXY', 'GLD'], start_date=None, end_date=None):
    """Adiciona variáveis exógenas ao DataFrame conforme documento de qualidade.
    
    Args:
        df (pd.DataFrame): DataFrame principal com dados OHLCV.
        use_comprehensive_collector (bool): Se True, usa o ExogenousDataCollector completo.
        symbols (list): Lista de símbolos para variáveis exógenas (usado apenas se use_comprehensive_collector=False).
        start_date (str): Data de início para coleta de dados exógenos.
        end_date (str): Data de fim para coleta de dados exógenos.
        
    Returns:
        pd.DataFrame: DataFrame com variáveis exógenas adicionadas.
    """
    # Verificar se o DataFrame é válido
    if df is None or df.empty:
        print("Erro: DataFrame vazio ou None.")
        return df
    
    # Criar cópia para não modificar o original
    df_result = df.copy()
    
    # Definir datas se não fornecidas
    if start_date is None:
        start_date = df_result.index.min().strftime('%Y-%m-%d')
    if end_date is None:
        end_date = df_result.index.max().strftime('%Y-%m-%d')
    
    if use_comprehensive_collector:
        try:
            # Importar o coletor de dados exógenos
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data_ingestion'))
            from exogenous_data import ExogenousDataCollector
            
            print("Usando ExogenousDataCollector para coleta abrangente de dados exógenos...")
            
            # Criar instância do coletor
            collector = ExogenousDataCollector()
            
            # Coletar dados exógenos consolidados
            exogenous_data = collector.consolidate_exogenous_data(start_date, end_date)
            
            if not exogenous_data.empty:
                # Sincronizar com dados do ativo principal
                df_sync, exog_sync = collector.synchronize_with_asset_data(df_result, exogenous_data)
                
                # Combinar os dados
                df_result = pd.concat([df_sync, exog_sync], axis=1)
                
                print(f"Dados exógenos abrangentes adicionados. Shape final: {df_result.shape}")
            else:
                print("Aviso: Nenhum dado exógeno foi coletado. Usando método simplificado...")
                use_comprehensive_collector = False
                
        except ImportError as e:
            print(f"Erro ao importar ExogenousDataCollector: {e}")
            print("Usando método simplificado de coleta de dados exógenos...")
            use_comprehensive_collector = False
        except Exception as e:
            print(f"Erro no ExogenousDataCollector: {e}")
            print("Usando método simplificado de coleta de dados exógenos...")
            use_comprehensive_collector = False
    
    # Método simplificado (fallback)
    if not use_comprehensive_collector:
        print(f"Coletando dados exógenos simplificados de {start_date} até {end_date}...")
        
        # Coletar dados para cada símbolo
        for symbol in symbols:
            try:
                print(f"Coletando dados para {symbol}...")
                
                # Baixar dados do Yahoo Finance
                ticker = yf.Ticker(symbol)
                exog_data = ticker.history(start=start_date, end=end_date)
                
                if exog_data.empty:
                    print(f"Aviso: Nenhum dado encontrado para {symbol}")
                    continue
                
                # Garantir que o índice seja tz-naive
                if exog_data.index.tz is not None:
                    exog_data.index = exog_data.index.tz_localize(None)
                
                # Renomear colunas para incluir o símbolo
                exog_data.columns = [f"{symbol}_{col}" for col in exog_data.columns]
                
                # Calcular indicadores específicos para cada variável exógena
                if symbol == 'SPY':  # S&P 500 ETF
                    exog_data[f'{symbol}_Return'] = exog_data[f'{symbol}_Close'].pct_change()
                    exog_data[f'{symbol}_SMA_20'] = ta.trend.sma_indicator(exog_data[f'{symbol}_Close'], window=20)
                    exog_data[f'{symbol}_RSI'] = ta.momentum.rsi(exog_data[f'{symbol}_Close'], window=14)
                    exog_data[f'{symbol}_Volatility'] = exog_data[f'{symbol}_Return'].rolling(window=20).std() * np.sqrt(252)
                    
                elif symbol == 'QQQ':  # NASDAQ ETF
                    exog_data[f'{symbol}_Return'] = exog_data[f'{symbol}_Close'].pct_change()
                    exog_data[f'{symbol}_SMA_10'] = ta.trend.sma_indicator(exog_data[f'{symbol}_Close'], window=10)
                    exog_data[f'{symbol}_Momentum'] = exog_data[f'{symbol}_Close'] - exog_data[f'{symbol}_Close'].shift(10)
                    
                elif symbol == 'VIX':  # Volatility Index
                    exog_data[f'{symbol}_Change'] = exog_data[f'{symbol}_Close'].diff()
                    exog_data[f'{symbol}_SMA_5'] = ta.trend.sma_indicator(exog_data[f'{symbol}_Close'], window=5)
                    exog_data[f'{symbol}_High_Low'] = (exog_data[f'{symbol}_Close'] > 20).astype(int)  # VIX > 20 indica alta volatilidade
                    exog_data[f'{symbol}_Extreme'] = (exog_data[f'{symbol}_Close'] > 30).astype(int)  # VIX > 30 indica volatilidade extrema
                    
                elif symbol == 'DXY':  # US Dollar Index
                    exog_data[f'{symbol}_Return'] = exog_data[f'{symbol}_Close'].pct_change()
                    exog_data[f'{symbol}_SMA_20'] = ta.trend.sma_indicator(exog_data[f'{symbol}_Close'], window=20)
                    exog_data[f'{symbol}_Strength'] = (exog_data[f'{symbol}_Close'] > exog_data[f'{symbol}_SMA_20']).astype(int)
                    
                elif symbol == 'GLD':  # Gold ETF
                    exog_data[f'{symbol}_Return'] = exog_data[f'{symbol}_Close'].pct_change()
                    exog_data[f'{symbol}_SMA_50'] = ta.trend.sma_indicator(exog_data[f'{symbol}_Close'], window=50)
                    exog_data[f'{symbol}_Trend'] = (exog_data[f'{symbol}_Close'] > exog_data[f'{symbol}_SMA_50']).astype(int)
                
                # Fazer merge com o DataFrame principal usando o índice (data)
                df_result = df_result.join(exog_data, how='left')
                
                print(f"Dados de {symbol} adicionados com sucesso.")
                
            except Exception as e:
                print(f"Erro ao coletar dados para {symbol}: {str(e)}")
                continue
        
        # === VARIÁVEIS EXÓGENAS DERIVADAS ===
        
        # Correlações com o ativo principal
        if 'SPY_Return' in df_result.columns and 'Returns' in df_result.columns:
            df_result['SPY_Correlation'] = df_result['Returns'].rolling(window=30).corr(df_result['SPY_Return'])
        
        if 'QQQ_Return' in df_result.columns and 'Returns' in df_result.columns:
            df_result['QQQ_Correlation'] = df_result['Returns'].rolling(window=30).corr(df_result['QQQ_Return'])
        
        # Spread entre diferentes índices
        if 'SPY_Close' in df_result.columns and 'QQQ_Close' in df_result.columns:
            df_result['SPY_QQQ_Spread'] = df_result['SPY_Close'] / df_result['QQQ_Close']
        
        # Indicador de risco (baseado no VIX)
        if 'VIX_Close' in df_result.columns:
            df_result['Market_Risk_Level'] = pd.cut(df_result['VIX_Close'], 
                                                   bins=[0, 15, 20, 30, 100], 
                                                   labels=[0, 1, 2, 3]).astype(float)
        
        # Força do dólar relativa
        if 'DXY_Close' in df_result.columns:
            df_result['DXY_Percentile'] = df_result['DXY_Close'].rolling(window=252).rank(pct=True)
        
        print(f"Variáveis exógenas simplificadas adicionadas. Shape final: {df_result.shape}")
    
    # Preencher valores NaN com forward fill e depois backward fill
    df_result = df_result.fillna(method='ffill').fillna(method='bfill')
    
    return df_result


def create_target_variables(df, prediction_horizon=1):
    """Cria variáveis alvo para previsão de preços futuros.
    
    Args:
        df (pd.DataFrame): DataFrame com dados OHLCV.
        prediction_horizon (int): Horizonte de previsão em dias.
        
    Returns:
        pd.DataFrame: DataFrame com variáveis alvo adicionadas.
    """
    # Verificar se o DataFrame é válido
    if df is None or df.empty:
        print("Erro: DataFrame vazio ou None.")
        return df
    
    # Verificar se a coluna necessária existe
    if 'Adj Close' not in df.columns:
        print("Erro: DataFrame não contém a coluna 'Adj Close'.")
        return df
    
    # Criar cópia para não modificar o original
    df_result = df.copy()
    
    # Criar variáveis alvo para diferentes horizontes de previsão
    for i in range(1, prediction_horizon + 1):
        # Preço futuro
        df_result[f'Target_Price_{i}d'] = df_result['Adj Close'].shift(-i)
        
        # Retorno futuro
        df_result[f'Target_Return_{i}d'] = df_result['Adj Close'].pct_change(-i)
        
        # Direção futura (1 se subir, 0 se descer)
        df_result[f'Target_Direction_{i}d'] = (df_result[f'Target_Return_{i}d'] > 0).astype(int)
    
    return df_result

def add_cyclical_features(df):
    """Adiciona features cíclicas baseadas em data/hora.
    
    Args:
        df (pd.DataFrame): DataFrame com índice de data/hora.
        
    Returns:
        pd.DataFrame: DataFrame com features cíclicas adicionadas.
    """
    # Verificar se o DataFrame é válido
    if df is None or df.empty:
        print("Erro: DataFrame vazio ou None.")
        return df
    
    # Verificar se o índice é datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        print("Erro: Índice do DataFrame não é do tipo DatetimeIndex.")
        return df
    
    # Criar cópia para não modificar o original
    df_result = df.copy()
    
    # Extrair componentes de data/hora
    df_result['Day_of_Week'] = df_result.index.dayofweek
    df_result['Day_of_Month'] = df_result.index.day
    df_result['Month'] = df_result.index.month
    df_result['Quarter'] = df_result.index.quarter
    df_result['Year'] = df_result.index.year
    
    # Converter para features cíclicas usando seno e cosseno
    # Dia da semana (0-6)
    df_result['Day_of_Week_sin'] = np.sin(2 * np.pi * df_result['Day_of_Week'] / 7)
    df_result['Day_of_Week_cos'] = np.cos(2 * np.pi * df_result['Day_of_Week'] / 7)
    
    # Dia do mês (1-31)
    df_result['Day_of_Month_sin'] = np.sin(2 * np.pi * df_result['Day_of_Month'] / 31)
    df_result['Day_of_Month_cos'] = np.cos(2 * np.pi * df_result['Day_of_Month'] / 31)
    
    # Mês (1-12)
    df_result['Month_sin'] = np.sin(2 * np.pi * df_result['Month'] / 12)
    df_result['Month_cos'] = np.cos(2 * np.pi * df_result['Month'] / 12)
    
    # Remover colunas intermediárias
    df_result = df_result.drop(['Day_of_Week', 'Day_of_Month', 'Month'], axis=1)
    
    return df_result

if __name__ == "__main__":
    # Exemplo de uso das funções implementadas
    import yfinance as yf
    
    # Baixar dados de exemplo
    ticker = "AAPL"
    print(f"Baixando dados para {ticker}...")
    data = yf.download(ticker, start="2023-01-01", end="2023-12-31")
    
    # Corrigir índices multi-level do yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    
    print(f"Dados originais: {data.shape}")
    print(f"Colunas disponíveis: {list(data.columns)}")
    
    # === DEMONSTRAÇÃO DOS INDICADORES TÉCNICOS CORRETOS ===
    print("\n=== Adicionando Indicadores Técnicos (Documento de Qualidade) ===")
    data_with_indicators = add_technical_indicators(data, include_legacy=False)
    print(f"Com indicadores técnicos: {data_with_indicators.shape}")
    
    # Mostrar indicadores adicionados
    new_indicators = [col for col in data_with_indicators.columns if col not in data.columns]
    print(f"\nIndicadores técnicos adicionados ({len(new_indicators)}):") 
    for indicator in new_indicators[:10]:  # Mostrar apenas os primeiros 10
        print(f"- {indicator}")
    if len(new_indicators) > 10:
        print(f"... e mais {len(new_indicators) - 10} indicadores")
    
    # === DEMONSTRAÇÃO DAS VARIÁVEIS EXÓGENAS ===
    print("\n=== Adicionando Variáveis Exógenas ===")
    try:
        data_with_exogenous = add_exogenous_variables(
            data_with_indicators, 
            symbols=['SPY', 'QQQ', 'VIX'],  # Usar menos símbolos para exemplo
            start_date="2023-01-01",
            end_date="2023-12-31"
        )
        print(f"Com variáveis exógenas: {data_with_exogenous.shape}")
        
        # Mostrar variáveis exógenas adicionadas
        exogenous_vars = [col for col in data_with_exogenous.columns if col not in data_with_indicators.columns]
        print(f"\nVariáveis exógenas adicionadas ({len(exogenous_vars)}):") 
        for var in exogenous_vars[:10]:  # Mostrar apenas as primeiras 10
            print(f"- {var}")
        if len(exogenous_vars) > 10:
            print(f"... e mais {len(exogenous_vars) - 10} variáveis")
        
        data_final = data_with_exogenous
    except Exception as e:
        print(f"Erro ao adicionar variáveis exógenas: {e}")
        print("Continuando sem variáveis exógenas...")
        data_final = data_with_indicators
    
    # === CRIAR VARIÁVEIS ALVO ===
    print("\n=== Criando Variáveis Alvo ===")
    data_with_targets = create_target_variables(data_final, prediction_horizon=5)
    print(f"Com variáveis alvo: {data_with_targets.shape}")
    
    # === ADICIONAR FEATURES CÍCLICAS ===
    print("\n=== Adicionando Features Cíclicas ===")
    data_complete = add_cyclical_features(data_with_targets)
    print(f"Dataset completo: {data_complete.shape}")
    
    # === RESUMO FINAL ===
    print("\n=== RESUMO FINAL ===")
    print(f"Dataset original: {data.shape}")
    print(f"Dataset final: {data_complete.shape}")
    print(f"Features adicionadas: {data_complete.shape[1] - data.shape[1]}")
    
    # Verificar conformidade com documento de qualidade
    required_indicators = [
        'BB_Upper', 'BB_Lower', 'BB_Position',  # Bollinger Bands
        'Stoch_K', 'Stoch_D',                   # Stochastic
        'Williams_R',                           # Williams %R
        'CCI',                                  # CCI
        'ADX', 'ADX_Pos', 'ADX_Neg',           # ADX
        'OBV',                                  # OBV
        'ATR',                                  # ATR
        'ROC_10', 'ROC_20',                    # ROC
        'TRIX'                                  # TRIX
    ]
    
    print("\n=== VERIFICAÇÃO DE CONFORMIDADE ===")
    missing_indicators = [ind for ind in required_indicators if ind not in data_complete.columns]
    
    if not missing_indicators:
        print("✅ Todos os indicadores técnicos requeridos estão presentes!")
    else:
        print(f"❌ Indicadores faltando: {missing_indicators}")
    
    # Mostrar estatísticas básicas de alguns indicadores
    print("\n=== ESTATÍSTICAS DOS INDICADORES ===")
    key_indicators = ['BB_Position', 'Stoch_K', 'Williams_R', 'CCI', 'ADX']
    available_indicators = [ind for ind in key_indicators if ind in data_complete.columns]
    
    if available_indicators:
        print(data_complete[available_indicators].describe())
    
    print("\n=== EXEMPLO CONCLUÍDO ===")
    print("O dataset está pronto para ser usado com os modelos RNN!")
    print("Próximos passos:")
    print("1. Pré-processamento e normalização")
    print("2. Divisão em treino/validação/teste")
    print("3. Treinamento dos modelos RNN")
    print("4. Avaliação com as métricas implementadas")
    print("5. Aplicação das estratégias de trading")

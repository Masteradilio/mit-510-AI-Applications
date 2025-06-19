# -*- coding: utf-8 -*-
"""Módulo de ingestão e processamento inicial de dados."""

import pandas as pd
import yfinance as yf
import os
from datetime import datetime

def load_kaggle_csv(file_path, date_col='Date', date_format=None):
    """Carrega um arquivo CSV do Kaggle, tratando a coluna de data e garantindo tz-naive.

    Args:
        file_path (str): Caminho para o arquivo CSV.
        date_col (str): Nome da coluna de data.
        date_format (str, optional): Formato da data, se necessário. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame carregado (tz-naive index) ou None se o arquivo não existir.
    """
    if not os.path.exists(file_path):
        print(f"Erro: Arquivo não encontrado em {file_path}")
        return None
    try:
        df = pd.read_csv(file_path)
        # Tenta converter a coluna de data para datetime
        if date_format:
            df[date_col] = pd.to_datetime(df[date_col], format=date_format)
        else:
            df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        # Garantir que o índice seja tz-naive
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.sort_index(inplace=True)
        return df
    except Exception as e:
        print(f"Erro ao carregar ou processar o arquivo {file_path}: {e}")
        return None

def fetch_yfinance_data(ticker, start_date=None, end_date=None):
    """Busca dados históricos usando a biblioteca yfinance, garantindo tz-naive.

    Args:
        ticker (str): Símbolo do ativo (ex: 'BTC-USD', 'AAPL').
        start_date (str, optional): Data de início (YYYY-MM-DD). Defaults to None (início histórico).
        end_date (str, optional): Data de fim (YYYY-MM-DD). Defaults to None (data atual).

    Returns:
        pd.DataFrame: DataFrame com dados do yfinance (tz-naive index) ou None se ocorrer erro.
    """
    try:
        stock = yf.Ticker(ticker)
        # Usar auto_adjust=True para obter preços ajustados automaticamente
        hist = stock.history(start=start_date, end=end_date, auto_adjust=True)
        # Renomear colunas para padronização (yfinance já usa nomes padrão)
        # Garantir que o índice seja datetime e tz-naive
        hist.index = pd.to_datetime(hist.index)
        if hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)
        # Garantir colunas essenciais (yfinance geralmente já as tem)
        # 'Adj Close' não estará presente com auto_adjust=True
        return hist
    except Exception as e:
        print(f"Erro ao buscar dados do yfinance para {ticker}: {e}")
        return None

def consolidate_data(kaggle_df, yfinance_df, standard_cols):
    """Consolida dados do Kaggle e yfinance (ambos tz-naive), priorizando yfinance.

    Args:
        kaggle_df (pd.DataFrame): DataFrame com dados do Kaggle (tz-naive index).
        yfinance_df (pd.DataFrame): DataFrame com dados do yfinance (tz-naive index).
        standard_cols (list): Lista de colunas padrão esperadas.

    Returns:
        pd.DataFrame: DataFrame consolidado e padronizado.
    """
    # Garantir que ambos os dataframes tenham índices tz-naive antes de prosseguir
    if kaggle_df is not None and kaggle_df.index.tz is not None:
        print("Aviso: Removendo timezone do índice do Kaggle DF.")
        kaggle_df.index = kaggle_df.index.tz_localize(None)
    if yfinance_df is not None and yfinance_df.index.tz is not None:
        print("Aviso: Removendo timezone do índice do yfinance DF.")
        yfinance_df.index = yfinance_df.index.tz_localize(None)

    if kaggle_df is None and yfinance_df is None:
        print("Erro: Nenhum DataFrame fornecido para consolidação.")
        return None

    # Caso apenas um dos DFs exista
    if yfinance_df is None:
        print("Aviso: Dados do yfinance não disponíveis, usando apenas Kaggle.")
        # Adicionar 'Adj Close' se não existir
        if 'Adj Close' not in kaggle_df.columns and 'Close' in kaggle_df.columns:
             kaggle_df['Adj Close'] = kaggle_df['Close']
        # Garantir colunas padrão
        for col in standard_cols:
            if col not in kaggle_df.columns:
                kaggle_df[col] = pd.NA
        return kaggle_df[standard_cols].copy()

    if kaggle_df is None:
        print("Aviso: Dados do Kaggle não disponíveis, usando apenas yfinance.")
        # Adicionar 'Adj Close' se não existir (auto_adjust=True remove)
        if 'Adj Close' not in yfinance_df.columns and 'Close' in yfinance_df.columns:
             yfinance_df['Adj Close'] = yfinance_df['Close']
        # Garantir colunas padrão
        for col in standard_cols:
            if col not in yfinance_df.columns:
                yfinance_df[col] = pd.NA
        return yfinance_df[standard_cols].copy()

    # Ambos os DFs existem e são tz-naive
    # Adicionar 'Adj Close' se não existir no yfinance_df (caso auto_adjust=True)
    if 'Adj Close' not in yfinance_df.columns and 'Close' in yfinance_df.columns:
        yfinance_df['Adj Close'] = yfinance_df['Close']
    # Adicionar 'Adj Close' se não existir no kaggle_df
    if 'Adj Close' not in kaggle_df.columns and 'Close' in kaggle_df.columns:
        kaggle_df['Adj Close'] = kaggle_df['Close']

    # Concatenar: yfinance tem prioridade onde há sobreposição
    # Mantém yfinance e adiciona apenas as datas do Kaggle que não estão no yfinance
    combined_df = pd.concat([kaggle_df[~kaggle_df.index.isin(yfinance_df.index)], yfinance_df])

    # Ordenar pelo índice (data) - agora seguro, pois ambos são tz-naive
    combined_df.sort_index(inplace=True)

    # Remover duplicatas de índice, mantendo a última (que seria do yfinance em caso de sobreposição)
    combined_df = combined_df[~combined_df.index.duplicated(keep='last')]

    # Garantir que todas as colunas padrão existam, preenchendo com NaN se necessário
    for col in standard_cols:
        if col not in combined_df.columns:
            combined_df[col] = pd.NA

    # Selecionar e reordenar para colunas padrão
    combined_df = combined_df[standard_cols]

    return combined_df

def save_processed_data(df, output_path):
    """Salva o DataFrame processado em um arquivo CSV.

    Args:
        df (pd.DataFrame): DataFrame a ser salvo.
        output_path (str): Caminho do arquivo de saída.

    Returns:
        bool: True se salvo com sucesso, False caso contrário.
    """
    if df is None:
        print("Erro: Nenhum DataFrame para salvar.")
        return False
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path)
        print(f"Dados processados salvos em {output_path}")
        return True
    except Exception as e:
        print(f"Erro ao salvar dados processados em {output_path}: {e}")
        return False

def run_ingestion_pipeline(asset_name, kaggle_file, yfinance_ticker, data_dir):
    """Executa o pipeline completo de ingestão para um ativo.

    Args:
        asset_name (str): Nome do ativo (ex: 'btc', 'aapl').
        kaggle_file (str): Nome do arquivo CSV do Kaggle na pasta raw.
        yfinance_ticker (str): Ticker do ativo no yfinance.
        data_dir (str): Diretório base do projeto ('agile_capital_forecast').

    Returns:
        pd.DataFrame: DataFrame processado e consolidado.
    """
    raw_dir = os.path.join(data_dir, 'data', 'raw')
    processed_dir = os.path.join(data_dir, 'data', 'processed')
    kaggle_path = os.path.join(raw_dir, kaggle_file)
    output_path = os.path.join(processed_dir, f"{asset_name}_processed.csv")

    print(f"--- Iniciando pipeline de ingestão para {asset_name.upper()} ---")

    # 1. Carregar dados do Kaggle (garantido tz-naive pela função)
    print(f"Carregando dados do Kaggle: {kaggle_path}")
    kaggle_df = load_kaggle_csv(kaggle_path)

    # 2. Buscar dados do yfinance (garantido tz-naive pela função)
    start_yfinance = None
    if kaggle_df is not None and not kaggle_df.empty:
        # Usar a última data do Kaggle como início para yfinance
        start_yfinance = (kaggle_df.index.max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        print(f"Buscando dados do yfinance para {yfinance_ticker} a partir de {start_yfinance}")
    else:
        print(f"Buscando histórico completo do yfinance para {yfinance_ticker}")

    yfinance_df = fetch_yfinance_data(yfinance_ticker, start_date=start_yfinance)

    # 3. Consolidar dados (agora ambos tz-naive)
    print("Consolidando dados do Kaggle e yfinance...")
    # Definir colunas padrão (garantir que 'Adj Close' exista)
    standard_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    processed_df = consolidate_data(kaggle_df, yfinance_df, standard_cols)

    # 4. Salvar dados processados
    if processed_df is not None:
        save_processed_data(processed_df, output_path)
    else:
        print(f"Falha ao processar dados para {asset_name.upper()}")

    print(f"--- Pipeline de ingestão para {asset_name.upper()} concluído ---")
    return processed_df

# Exemplo de uso (pode ser chamado de outro script ou notebook)
if __name__ == '__main__':
    # Definir diretório base relativo à localização deste script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Assumindo que loader.py está em src/data_ingestion
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))

    # Executar para BTC
    btc_df = run_ingestion_pipeline(
        asset_name='btc',
        kaggle_file='BTC-USD From 2014 To Dec-2024.csv',
        yfinance_ticker='BTC-USD',
        data_dir=project_root
    )

    # Executar para AAPL
    aapl_kaggle_file = 'apple-stockprice-2014-2024.csv'
    aapl_df = run_ingestion_pipeline(
        asset_name='aapl',
        kaggle_file=aapl_kaggle_file,
        yfinance_ticker='AAPL',
        data_dir=project_root
    )

    print("\nDatasets processados:")
    if btc_df is not None:
        print("\nBTC:")
        print(btc_df.head())
        print(btc_df.tail())
        print(f"Shape: {btc_df.shape}")
    if aapl_df is not None:
        print("\nAAPL:")
        print(aapl_df.head())
        print(aapl_df.tail())
        print(f"Shape: {aapl_df.shape}")


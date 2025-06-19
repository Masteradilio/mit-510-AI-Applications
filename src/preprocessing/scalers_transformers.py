# -*- coding: utf-8 -*-
"""Módulo para transformação e normalização de dados."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def handle_missing_values(df, method='ffill'):
    """Trata valores ausentes no DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame com possíveis valores ausentes.
        method (str): Método para preenchimento ('ffill', 'bfill', 'zero', 'mean').
        
    Returns:
        pd.DataFrame: DataFrame com valores ausentes tratados.
    """
    if df is None or df.empty:
        print("Erro: DataFrame vazio ou None.")
        return df
    
    # Criar cópia para não modificar o original
    df_result = df.copy()
    
    # Verificar se há valores ausentes
    missing_count = df_result.isna().sum().sum()
    if missing_count > 0:
        print(f"Encontrados {missing_count} valores ausentes no DataFrame.")
        
        if method == 'ffill':
            # Forward fill (propaga o último valor válido)
            df_result = df_result.fillna(method='ffill')
            # Se ainda houver NaN no início, fazer backward fill
            df_result = df_result.fillna(method='bfill')
        elif method == 'bfill':
            # Backward fill (propaga o próximo valor válido)
            df_result = df_result.fillna(method='bfill')
            # Se ainda houver NaN no final, fazer forward fill
            df_result = df_result.fillna(method='ffill')
        elif method == 'zero':
            # Preencher com zeros
            df_result = df_result.fillna(0)
        elif method == 'mean':
            # Preencher com a média da coluna
            for col in df_result.columns:
                if df_result[col].isna().any():
                    col_mean = df_result[col].mean()
                    df_result[col] = df_result[col].fillna(col_mean)
        else:
            print(f"Método '{method}' não reconhecido. Usando 'ffill' como padrão.")
            df_result = df_result.fillna(method='ffill')
            df_result = df_result.fillna(method='bfill')
    
    # Verificar se ainda há valores ausentes
    remaining_missing = df_result.isna().sum().sum()
    if remaining_missing > 0:
        print(f"Atenção: Ainda restam {remaining_missing} valores ausentes após tratamento.")
    
    return df_result

def create_data_scaler(scaler_type='minmax'):
    """Cria um scaler para normalização de dados.
    
    Args:
        scaler_type (str): Tipo de scaler ('minmax' ou 'standard').
        
    Returns:
        object: Instância do scaler escolhido.
    """
    if scaler_type.lower() == 'minmax':
        return MinMaxScaler(feature_range=(0, 1))
    elif scaler_type.lower() == 'standard':
        return StandardScaler()
    else:
        print(f"Tipo de scaler '{scaler_type}' não reconhecido. Usando 'minmax' como padrão.")
        return MinMaxScaler(feature_range=(0, 1))

def scale_data(df, scaler=None, fit=True, columns=None):
    """Normaliza os dados usando o scaler fornecido ou criando um novo.
    
    Args:
        df (pd.DataFrame): DataFrame a ser normalizado.
        scaler (object, optional): Scaler pré-ajustado. Se None, cria um novo MinMaxScaler.
        fit (bool): Se True, ajusta o scaler aos dados. Se False, usa o scaler já ajustado.
        columns (list, optional): Lista de colunas para normalizar. Se None, usa todas as colunas numéricas.
        
    Returns:
        tuple: (DataFrame normalizado, scaler usado)
    """
    if df is None or df.empty:
        print("Erro: DataFrame vazio ou None.")
        return df, scaler
    
    # Criar cópia para não modificar o original
    df_result = df.copy()
    
    # Selecionar colunas para normalizar
    if columns is None:
        # Usar todas as colunas numéricas
        columns = df_result.select_dtypes(include=np.number).columns.tolist()
    else:
        # Verificar se as colunas existem
        for col in columns:
            if col not in df_result.columns:
                print(f"Aviso: Coluna '{col}' não encontrada no DataFrame.")
                columns.remove(col)
    
    if not columns:
        print("Erro: Nenhuma coluna válida para normalizar.")
        return df_result, scaler
    
    # Criar scaler se não fornecido
    if scaler is None:
        scaler = create_data_scaler('minmax')
    
    # Extrair os valores das colunas selecionadas
    values = df_result[columns].values
    
    # Ajustar o scaler e transformar os dados
    if fit:
        values_scaled = scaler.fit_transform(values)
    else:
        values_scaled = scaler.transform(values)
    
    # Substituir os valores originais pelos normalizados
    df_result[columns] = values_scaled
    
    return df_result, scaler

def create_sequences(data, seq_length, target_col_idx=0, forecast_horizon=1, step=1):
    """Cria sequências de entrada e saída para modelos de séries temporais.
    
    Args:
        data (np.ndarray): Array 2D com os dados normalizados.
        seq_length (int): Comprimento da sequência de entrada.
        target_col_idx (int): Índice da coluna alvo para previsão.
        forecast_horizon (int): Horizonte de previsão (número de passos futuros).
        step (int): Passo entre sequências consecutivas.
        
    Returns:
        tuple: (X, y) onde X são as sequências de entrada e y são os alvos.
    """
    X, y = [], []
    
    for i in range(0, len(data) - seq_length - forecast_horizon + 1, step):
        # Sequência de entrada
        seq_x = data[i:i+seq_length]
        X.append(seq_x)
        
        # Sequência de saída (apenas a coluna alvo)
        seq_y = data[i+seq_length:i+seq_length+forecast_horizon, target_col_idx]
        y.append(seq_y)
    
    return np.array(X), np.array(y)

def split_data(df, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, shuffle=False):
    """Divide os dados em conjuntos de treino, validação e teste.
    
    Args:
        df (pd.DataFrame): DataFrame a ser dividido.
        train_ratio (float): Proporção para o conjunto de treino.
        val_ratio (float): Proporção para o conjunto de validação.
        test_ratio (float): Proporção para o conjunto de teste.
        shuffle (bool): Se True, embaralha os dados antes da divisão.
        
    Returns:
        tuple: (df_train, df_val, df_test)
    """
    if df is None or df.empty:
        print("Erro: DataFrame vazio ou None.")
        return None, None, None
    
    # Verificar se as proporções somam 1
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        print(f"Aviso: As proporções somam {total_ratio}, não 1.0. Normalizando...")
        train_ratio = train_ratio / total_ratio
        val_ratio = val_ratio / total_ratio
        test_ratio = test_ratio / total_ratio
    
    # Número de amostras em cada conjunto
    n = len(df)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    # Criar índices
    indices = np.arange(n)
    if shuffle:
        np.random.shuffle(indices)
    
    # Dividir os dados
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size+val_size]
    test_indices = indices[train_size+val_size:]
    
    # Para séries temporais, geralmente mantemos a ordem cronológica
    if not shuffle:
        df_train = df.iloc[:train_size]
        df_val = df.iloc[train_size:train_size+val_size]
        df_test = df.iloc[train_size+val_size:]
    else:
        df_train = df.iloc[train_indices]
        df_val = df.iloc[val_indices]
        df_test = df.iloc[test_indices]
    
    print(f"Divisão dos dados: Treino={len(df_train)} ({train_ratio:.1%}), "
          f"Validação={len(df_val)} ({val_ratio:.1%}), "
          f"Teste={len(df_test)} ({test_ratio:.1%})")
    
    return df_train, df_val, df_test

# Exemplo de uso
if __name__ == "__main__":
    # Criar dados de exemplo
    dates = pd.date_range(start="2020-01-01", periods=100)
    data = {
        "Feature1": np.random.randn(100).cumsum(),
        "Feature2": np.random.randn(100).cumsum() * 2,
        "Target": np.random.randn(100).cumsum() + 100
    }
    df = pd.DataFrame(data, index=dates)
    
    # Adicionar alguns valores ausentes
    df.loc[10:15, "Feature1"] = np.nan
    
    # Tratar valores ausentes
    df_clean = handle_missing_values(df)
    print(f"Valores ausentes após tratamento: {df_clean.isna().sum().sum()}")
    
    # Normalizar os dados
    df_scaled, scaler = scale_data(df_clean)
    print(f"Range após normalização: Min={df_scaled.min().min():.4f}, Max={df_scaled.max().max():.4f}")
    
    # Dividir os dados
    df_train, df_val, df_test = split_data(df_scaled, shuffle=False)
    
    # Criar sequências para RNN
    X_train, y_train = create_sequences(df_train.values, seq_length=10, target_col_idx=2, forecast_horizon=5)
    print(f"Shape das sequências de treino: X={X_train.shape}, y={y_train.shape}")

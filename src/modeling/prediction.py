# -*- coding: utf-8 -*-
"""Módulo para carregar modelos, preparar dados e gerar previsões."""

import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

# Importar funções de pré-processamento necessárias
# Adicionar o diretório src ao path para garantir importações corretas
import sys
try:
    module_dir = os.path.dirname(__file__)
except NameError:
    module_dir = os.getcwd()

src_path = os.path.abspath(os.path.join(module_dir, "..", "..")) # Vai para a raiz do projeto
if src_path not in sys.path:
    sys.path.append(src_path)

from src.preprocessing import scalers_transformers

# Constantes (devem corresponder às usadas no treinamento)
SEQUENCE_LENGTH = 60
FORECAST_HORIZON = 14

def load_model(model_path):
    """Carrega um modelo Keras salvo."""
    if not os.path.exists(model_path):
        print(f"Erro: Arquivo do modelo não encontrado em {model_path}")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Modelo carregado de {model_path}")
        return model
    except Exception as e:
        print(f"Erro ao carregar modelo de {model_path}: {e}")
        return None

def load_scaler(scaler_path):
    """Carrega um scaler scikit-learn salvo."""
    if not os.path.exists(scaler_path):
        print(f"Erro: Arquivo do scaler não encontrado em {scaler_path}")
        return None
    try:
        scaler = joblib.load(scaler_path)
        print(f"Scaler carregado de {scaler_path}")
        return scaler
    except Exception as e:
        print(f"Erro ao carregar scaler de {scaler_path}: {e}")
        return None

def prepare_input_sequence(df, scaler):
    """Prepara a última sequência de dados para a previsão.

    Args:
        df (pd.DataFrame): DataFrame completo com dados históricos e indicadores.
        scaler (sklearn.preprocessing.Scaler): Scaler ajustado nos dados de treino.

    Returns:
        np.ndarray: Sequência escalonada pronta para o modelo (shape: 1, seq_len, n_features).
                    Retorna None se houver erro ou dados insuficientes.
    """
    if df is None or scaler is None:
        print("Erro: DataFrame ou scaler não fornecido para preparar sequência.")
        return None

    if len(df) < SEQUENCE_LENGTH:
        print(f"Erro: Dados insuficientes ({len(df)} linhas) para criar sequência de tamanho {SEQUENCE_LENGTH}.")
        return None

    # Pegar as últimas SEQUENCE_LENGTH linhas
    last_sequence_df = df.iloc[-SEQUENCE_LENGTH:]

    # Selecionar apenas colunas numéricas que o scaler espera
    try:
        expected_cols = scaler.feature_names_in_
        last_sequence_numeric = last_sequence_df[expected_cols]
    except AttributeError:
        print("Aviso: Scaler não possui `feature_names_in_`. Usando todas as colunas numéricas.")
        last_sequence_numeric = last_sequence_df.select_dtypes(include=np.number)
        if last_sequence_numeric.shape[1] != scaler.n_features_in_:
             print(f"Erro: Discrepância de features entre sequência ({last_sequence_numeric.shape[1]}) e scaler ({scaler.n_features_in_}).")
             return None
    except KeyError as e:
        print(f"Erro: Coluna esperada pelo scaler não encontrada no DataFrame: {e}")
        return None

    # Escalonar a sequência
    try:
        sequence_scaled = scaler.transform(last_sequence_numeric)
    except Exception as e:
        print(f"Erro ao aplicar scaler na sequência: {e}")
        return None

    # Adicionar dimensão do batch
    sequence_scaled = np.expand_dims(sequence_scaled, axis=0)

    return sequence_scaled

def generate_forecast(model, input_sequence):
    """Gera a previsão usando o modelo carregado e a sequência de entrada.

    Args:
        model (tf.keras.Model): Modelo RNN treinado.
        input_sequence (np.ndarray): Sequência de entrada escalonada (shape: 1, seq_len, n_features).

    Returns:
        np.ndarray: Previsão escalonada (shape: 1, forecast_horizon).
                    Retorna None se houver erro.
    """
    if model is None or input_sequence is None:
        print("Erro: Modelo ou sequência de entrada não fornecida para gerar previsão.")
        return None

    try:
        prediction_scaled = model.predict(input_sequence)
        return prediction_scaled
    except Exception as e:
        print(f"Erro durante a previsão do modelo: {e}")
        return None

def inverse_transform_forecast(prediction_scaled, scaler):
    """Desnormaliza a previsão escalonada.

    Args:
        prediction_scaled (np.ndarray): Previsão escalonada (shape: 1, forecast_horizon).
        scaler (sklearn.preprocessing.Scaler): Scaler usado no treinamento.

    Returns:
        np.ndarray: Previsão desnormalizada (shape: forecast_horizon,).
                    Retorna None se houver erro.
    """
    if prediction_scaled is None or scaler is None:
        print("Erro: Previsão escalonada ou scaler não fornecido para desnormalização.")
        return None

    try:
        num_features = scaler.n_features_in_
        forecast_horizon = prediction_scaled.shape[1]
        prediction_inv = np.zeros((1, forecast_horizon))

        # Assumimos que a coluna alvo (Adj Close) é a primeira (índice 0) no scaler
        target_col_index = 0

        for i in range(forecast_horizon):
            dummy_pred = np.zeros((1, num_features))
            dummy_pred[0, target_col_index] = prediction_scaled[0, i]
            prediction_inv[0, i] = scaler.inverse_transform(dummy_pred)[0, target_col_index]

        return prediction_inv.flatten()
    except Exception as e:
        print(f"Erro durante a desnormalização da previsão: {e}")
        return None

# Exemplo de pipeline de previsão (pode ser chamado pelo app Streamlit)
def run_prediction_pipeline(df, model_path, scaler_path):
    """Executa o pipeline completo de previsão.

    Args:
        df (pd.DataFrame): DataFrame histórico completo.
        model_path (str): Caminho para o arquivo do modelo .h5.
        scaler_path (str): Caminho para o arquivo do scaler .joblib.

    Returns:
        np.ndarray: Array com os valores da previsão desnormalizada (shape: forecast_horizon,).
                    Retorna None se alguma etapa falhar.
    """
    model = load_model(model_path)
    scaler = load_scaler(scaler_path)

    if model is None or scaler is None:
        return None

    input_sequence = prepare_input_sequence(df, scaler)
    if input_sequence is None:
        return None

    prediction_scaled = generate_forecast(model, input_sequence)
    if prediction_scaled is None:
        return None

    prediction_final = inverse_transform_forecast(prediction_scaled, scaler)
    return prediction_final

# Exemplo de uso (para teste)
if __name__ == "__main__":
    # Caminhos de exemplo (ajuste conforme necessário)
    example_models_path = os.path.join(src_path, "..", "models") # Ajuste para a pasta models real
    example_data_path = os.path.join(src_path, "..", "data", "processed") # Ajuste para a pasta data/processed real

    asset = "btc" # ou "aapl"
    model_type = "lstm" # ou "gru", "simplernn"

    example_model_file = os.path.join(example_models_path, f"{asset}_{model_type}_best.h5")
    example_scaler_file = os.path.join(example_models_path, f"{asset}_scaler.joblib")
    example_data_file = os.path.join(example_data_path, f"{asset}_processed.csv")

    if not os.path.exists(example_data_file):
        print(f"Arquivo de dados de exemplo não encontrado: {example_data_file}")
    else:
        df_example = pd.read_csv(example_data_file, index_col="Date", parse_dates=True)
        # Adicionar indicadores se necessário (simulando o estado após o notebook)
        if "SMA_50" not in df_example.columns:
             from src.preprocessing import feature_engineering
             df_example = feature_engineering.add_technical_indicators(df_example.copy())
             df_example = scalers_transformers.handle_missing_values(df_example.copy())

        print(f"\n--- Executando Pipeline de Previsão para {asset.upper()} com {model_type.upper()} ---")
        forecast = run_prediction_pipeline(df_example, example_model_file, example_scaler_file)

        if forecast is not None:
            print(f"\nPrevisão para os próximos {FORECAST_HORIZON} dias:")
            last_date = df_example.index[-1]
            forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=FORECAST_HORIZON)
            forecast_series = pd.Series(forecast, index=forecast_dates)
            print(forecast_series)
        else:
            print("\nFalha ao executar o pipeline de previsão.")

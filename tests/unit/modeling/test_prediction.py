# -*- coding: utf-8 -*-
"""Testes unitários para o módulo prediction.py."""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from unittest.mock import patch, MagicMock

# Adicionar diretório raiz ao path
test_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(test_dir, "..", "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Importar o módulo a ser testado
from src.modeling import prediction

class TestPrediction(unittest.TestCase):
    """Testes para o módulo prediction.py."""
    
    def setUp(self):
        """Configuração inicial para os testes."""
        # Criar dados de teste
        self.test_dates = pd.date_range(start="2023-01-01", periods=100)
        self.test_df = pd.DataFrame({
            "Adj Close": 100 + np.random.randn(100).cumsum(),
            "Volume": np.random.randint(10000, 50000, 100),
            "SMA_50": (100 + np.random.randn(100).cumsum()) * 0.98,
            "RSI": np.random.uniform(30, 70, 100)
        }, index=self.test_dates)
        
        # Criar diretório temporário para modelos de teste
        self.test_models_dir = os.path.join(project_root, "tests", "temp_models")
        os.makedirs(self.test_models_dir, exist_ok=True)
        
        # Criar um modelo simples para teste
        self.test_model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(60, 4)),
            tf.keras.layers.Dense(14)
        ])
        self.test_model.compile(optimizer='adam', loss='mse')
        
        # Criar um scaler simples para teste
        from sklearn.preprocessing import MinMaxScaler
        self.test_scaler = MinMaxScaler()
        self.test_scaler.fit(self.test_df.values)
        
        # Salvar modelo e scaler para testes
        self.test_model_path = os.path.join(self.test_models_dir, "test_model.h5")
        self.test_scaler_path = os.path.join(self.test_models_dir, "test_scaler.joblib")
        
        self.test_model.save(self.test_model_path)
        joblib.dump(self.test_scaler, self.test_scaler_path)
        
        # Constantes
        self.SEQUENCE_LENGTH = 60
        self.FORECAST_HORIZON = 14
    
    def tearDown(self):
        """Limpeza após os testes."""
        # Remover arquivos temporários
        if os.path.exists(self.test_model_path):
            os.remove(self.test_model_path)
        if os.path.exists(self.test_scaler_path):
            os.remove(self.test_scaler_path)
        
        # Remover diretório temporário se estiver vazio
        try:
            os.rmdir(self.test_models_dir)
        except OSError:
            pass  # Diretório não está vazio ou não existe
    
    def test_load_model(self):
        """Testa a função load_model."""
        # Testar carregamento de modelo existente
        model = prediction.load_model(self.test_model_path)
        self.assertIsNotNone(model, "Modelo deveria ser carregado com sucesso")
        
        # Testar carregamento de modelo inexistente
        model = prediction.load_model("modelo_inexistente.h5")
        self.assertIsNone(model, "Modelo inexistente deveria retornar None")
    
    def test_load_scaler(self):
        """Testa a função load_scaler."""
        # Testar carregamento de scaler existente
        scaler = prediction.load_scaler(self.test_scaler_path)
        self.assertIsNotNone(scaler, "Scaler deveria ser carregado com sucesso")
        
        # Testar carregamento de scaler inexistente
        scaler = prediction.load_scaler("scaler_inexistente.joblib")
        self.assertIsNone(scaler, "Scaler inexistente deveria retornar None")
    
    def test_prepare_input_sequence(self):
        """Testa a função prepare_input_sequence."""
        # Testar com dados válidos
        with patch.object(prediction, 'SEQUENCE_LENGTH', self.SEQUENCE_LENGTH):
            input_seq = prediction.prepare_input_sequence(self.test_df, self.test_scaler)
            self.assertIsNotNone(input_seq, "Sequência de entrada deveria ser criada com sucesso")
            self.assertEqual(input_seq.shape, (1, self.SEQUENCE_LENGTH, self.test_df.shape[1]), 
                            "Shape da sequência de entrada está incorreto")
            
            # Testar com DataFrame None
            input_seq = prediction.prepare_input_sequence(None, self.test_scaler)
            self.assertIsNone(input_seq, "DataFrame None deveria retornar None")
            
            # Testar com scaler None
            input_seq = prediction.prepare_input_sequence(self.test_df, None)
            self.assertIsNone(input_seq, "Scaler None deveria retornar None")
            
            # Testar com dados insuficientes
            small_df = self.test_df.iloc[:30]  # Menos que SEQUENCE_LENGTH
            input_seq = prediction.prepare_input_sequence(small_df, self.test_scaler)
            self.assertIsNone(input_seq, "Dados insuficientes deveriam retornar None")
    
    def test_generate_forecast(self):
        """Testa a função generate_forecast."""
        # Criar sequência de entrada mockada
        mock_input = np.random.rand(1, self.SEQUENCE_LENGTH, 4)
        
        # Testar com modelo válido
        forecast = prediction.generate_forecast(self.test_model, mock_input)
        self.assertIsNotNone(forecast, "Previsão deveria ser gerada com sucesso")
        self.assertEqual(forecast.shape, (1, self.FORECAST_HORIZON), 
                        "Shape da previsão está incorreto")
        
        # Testar com modelo None
        forecast = prediction.generate_forecast(None, mock_input)
        self.assertIsNone(forecast, "Modelo None deveria retornar None")
        
        # Testar com sequência None
        forecast = prediction.generate_forecast(self.test_model, None)
        self.assertIsNone(forecast, "Sequência None deveria retornar None")
    
    def test_inverse_transform_forecast(self):
        """Testa a função inverse_transform_forecast."""
        # Criar previsão mockada
        mock_prediction = np.random.rand(1, self.FORECAST_HORIZON)
        
        # Testar com dados válidos
        with patch.object(prediction, 'FORECAST_HORIZON', self.FORECAST_HORIZON):
            forecast_inv = prediction.inverse_transform_forecast(mock_prediction, self.test_scaler)
            self.assertIsNotNone(forecast_inv, "Previsão desnormalizada deveria ser gerada com sucesso")
            self.assertEqual(forecast_inv.shape, (self.FORECAST_HORIZON,), 
                            "Shape da previsão desnormalizada está incorreto")
            
            # Testar com previsão None
            forecast_inv = prediction.inverse_transform_forecast(None, self.test_scaler)
            self.assertIsNone(forecast_inv, "Previsão None deveria retornar None")
            
            # Testar com scaler None
            forecast_inv = prediction.inverse_transform_forecast(mock_prediction, None)
            self.assertIsNone(forecast_inv, "Scaler None deveria retornar None")
    
    def test_run_prediction_pipeline(self):
        """Testa a função run_prediction_pipeline."""
        # Mockando as funções internas para isolar o teste
        with patch('src.modeling.prediction.load_model') as mock_load_model, \
             patch('src.modeling.prediction.load_scaler') as mock_load_scaler, \
             patch('src.modeling.prediction.prepare_input_sequence') as mock_prepare, \
             patch('src.modeling.prediction.generate_forecast') as mock_generate, \
             patch('src.modeling.prediction.inverse_transform_forecast') as mock_inverse:
            
            # Configurar mocks
            mock_model = MagicMock()
            mock_scaler = MagicMock()
            mock_input_seq = np.random.rand(1, 60, 4)
            mock_prediction = np.random.rand(1, 14)
            mock_forecast = np.random.rand(14)
            
            mock_load_model.return_value = mock_model
            mock_load_scaler.return_value = mock_scaler
            mock_prepare.return_value = mock_input_seq
            mock_generate.return_value = mock_prediction
            mock_inverse.return_value = mock_forecast
            
            # Testar pipeline completo
            result = prediction.run_prediction_pipeline(self.test_df, "model_path", "scaler_path")
            
            # Verificar se todas as funções foram chamadas corretamente
            mock_load_model.assert_called_once_with("model_path")
            mock_load_scaler.assert_called_once_with("scaler_path")
            mock_prepare.assert_called_once_with(self.test_df, mock_scaler)
            mock_generate.assert_called_once_with(mock_model, mock_input_seq)
            mock_inverse.assert_called_once_with(mock_prediction, mock_scaler)
            
            # Verificar resultado
            self.assertEqual(result.tolist(), mock_forecast.tolist(), 
                            "Resultado do pipeline deveria ser igual à previsão desnormalizada")
            
            # Testar falha em load_model
            mock_load_model.return_value = None
            result = prediction.run_prediction_pipeline(self.test_df, "model_path", "scaler_path")
            self.assertIsNone(result, "Falha em load_model deveria retornar None")
            
            # Testar falha em load_scaler
            mock_load_model.return_value = mock_model
            mock_load_scaler.return_value = None
            result = prediction.run_prediction_pipeline(self.test_df, "model_path", "scaler_path")
            self.assertIsNone(result, "Falha em load_scaler deveria retornar None")

if __name__ == '__main__':
    unittest.main()

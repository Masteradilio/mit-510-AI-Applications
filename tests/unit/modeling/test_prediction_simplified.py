# -*- coding: utf-8 -*-
"""Testes unitários simplificados para o módulo prediction.py."""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd

# Adicionar diretório raiz ao path
test_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(test_dir, "..", "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Mockar tensorflow e joblib para evitar dependências pesadas
sys.modules['tensorflow'] = MagicMock()
sys.modules['joblib'] = MagicMock()

# Importar o módulo a ser testado com mocks
from src.modeling import prediction

# Sobrescrever constantes para testes
prediction.SEQUENCE_LENGTH = 10
prediction.FORECAST_HORIZON = 5

class TestPredictionSimplified(unittest.TestCase):
    """Testes simplificados para o módulo prediction.py."""
    
    def setUp(self):
        """Configuração inicial para os testes."""
        # Criar dados de teste
        self.test_dates = pd.date_range(start="2023-01-01", periods=20)
        self.test_df = pd.DataFrame({
            "Adj Close": 100 + np.random.randn(20).cumsum(),
            "Volume": np.random.randint(10000, 50000, 20),
            "Feature1": np.random.randn(20),
            "Feature2": np.random.randn(20)
        }, index=self.test_dates)
        
        # Mock para scaler
        self.mock_scaler = MagicMock()
        self.mock_scaler.n_features_in_ = 4
        self.mock_scaler.transform.return_value = np.random.rand(10, 4)
        self.mock_scaler.inverse_transform.return_value = np.random.rand(1, 4) * 100
        
        # Mock para modelo
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = np.random.rand(1, 5)  # (batch_size, forecast_horizon)
    
    def test_prepare_input_sequence(self):
        """Testa a função prepare_input_sequence com mocks."""
        # Configurar feature_names_in_ para o scaler
        self.mock_scaler.feature_names_in_ = self.test_df.columns.tolist()
        
        # Testar com dados válidos
        input_seq = prediction.prepare_input_sequence(self.test_df, self.mock_scaler)
        self.assertIsNotNone(input_seq, "Sequência de entrada deveria ser criada com sucesso")
        self.mock_scaler.transform.assert_called_once()
        
        # Testar com DataFrame None
        self.mock_scaler.transform.reset_mock()
        input_seq = prediction.prepare_input_sequence(None, self.mock_scaler)
        self.assertIsNone(input_seq, "DataFrame None deveria retornar None")
        self.mock_scaler.transform.assert_not_called()
        
        # Testar com scaler None
        input_seq = prediction.prepare_input_sequence(self.test_df, None)
        self.assertIsNone(input_seq, "Scaler None deveria retornar None")
    
    def test_generate_forecast(self):
        """Testa a função generate_forecast com mocks."""
        # Criar sequência de entrada mockada
        mock_input = np.random.rand(1, 10, 4)
        
        # Testar com modelo válido
        forecast = prediction.generate_forecast(self.mock_model, mock_input)
        self.assertIsNotNone(forecast, "Previsão deveria ser gerada com sucesso")
        self.mock_model.predict.assert_called_once_with(mock_input)
        
        # Testar com modelo None
        self.mock_model.predict.reset_mock()
        forecast = prediction.generate_forecast(None, mock_input)
        self.assertIsNone(forecast, "Modelo None deveria retornar None")
        self.mock_model.predict.assert_not_called()
        
        # Testar com sequência None
        forecast = prediction.generate_forecast(self.mock_model, None)
        self.assertIsNone(forecast, "Sequência None deveria retornar None")
        self.mock_model.predict.assert_not_called()
    
    def test_inverse_transform_forecast(self):
        """Testa a função inverse_transform_forecast com mocks."""
        # Criar previsão mockada
        mock_prediction = np.random.rand(1, 5)
        
        # Testar com dados válidos
        forecast_inv = prediction.inverse_transform_forecast(mock_prediction, self.mock_scaler)
        self.assertIsNotNone(forecast_inv, "Previsão desnormalizada deveria ser gerada com sucesso")
        self.assertEqual(len(forecast_inv), 5, "Previsão desnormalizada deveria ter tamanho 5")
        self.assertEqual(self.mock_scaler.inverse_transform.call_count, 5)
        
        # Testar com previsão None
        self.mock_scaler.inverse_transform.reset_mock()
        forecast_inv = prediction.inverse_transform_forecast(None, self.mock_scaler)
        self.assertIsNone(forecast_inv, "Previsão None deveria retornar None")
        self.mock_scaler.inverse_transform.assert_not_called()
        
        # Testar com scaler None
        forecast_inv = prediction.inverse_transform_forecast(mock_prediction, None)
        self.assertIsNone(forecast_inv, "Scaler None deveria retornar None")
    
    def test_run_prediction_pipeline(self):
        """Testa a função run_prediction_pipeline com mocks."""
        # Mockando as funções internas
        with patch('src.modeling.prediction.load_model', return_value=self.mock_model), \
             patch('src.modeling.prediction.load_scaler', return_value=self.mock_scaler), \
             patch('src.modeling.prediction.prepare_input_sequence', return_value=np.random.rand(1, 10, 4)), \
             patch('src.modeling.prediction.generate_forecast', return_value=np.random.rand(1, 5)), \
             patch('src.modeling.prediction.inverse_transform_forecast', return_value=np.random.rand(5)):
            
            # Testar pipeline completo
            result = prediction.run_prediction_pipeline(self.test_df, "model_path", "scaler_path")
            self.assertIsNotNone(result, "Pipeline deveria retornar resultado válido")
            self.assertEqual(len(result), 5, "Resultado deveria ter tamanho 5")

if __name__ == '__main__':
    unittest.main()

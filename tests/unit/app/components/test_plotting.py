# -*- coding: utf-8 -*-
"""Testes unitários para o módulo plotting.py."""

import os
import sys
import unittest
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Adicionar diretório raiz ao path
test_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(test_dir, "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Importar o módulo a ser testado
from src.app.components import plotting

class TestPlotting(unittest.TestCase):
    """Testes para o módulo plotting.py."""
    
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
        
        # Criar dados de previsão
        self.forecast_dates = pd.date_range(start=self.test_dates[-1] + pd.Timedelta(days=1), periods=14)
        self.forecast_values = self.test_df["Adj Close"].iloc[-1] * (1 + np.random.randn(14)*0.02).cumsum()
        
        # Criar dados de retornos
        self.returns_dict = {
            "Long-Only (Forecast)": pd.Series(np.random.randn(100)*0.01, index=self.test_dates),
            "Buy-and-Hold": pd.Series(np.random.randn(100)*0.008, index=self.test_dates)
        }
    
    def test_plot_historical_data(self):
        """Testa a função plot_historical_data."""
        # Testar com dados válidos
        fig = plotting.plot_historical_data(self.test_df, "BTC-USD")
        self.assertIsInstance(fig, go.Figure, "Deveria retornar um objeto Figure do Plotly")
        self.assertEqual(len(fig.data), 2, "Deveria ter 2 traces (preço e volume)")
        
        # Testar com DataFrame vazio
        fig = plotting.plot_historical_data(pd.DataFrame(), "BTC-USD")
        self.assertIsInstance(fig, go.Figure, "Deveria retornar um objeto Figure do Plotly mesmo com DataFrame vazio")
        
        # Testar com DataFrame None
        fig = plotting.plot_historical_data(None, "BTC-USD")
        self.assertIsInstance(fig, go.Figure, "Deveria retornar um objeto Figure do Plotly mesmo com DataFrame None")
    
    def test_plot_technical_indicators(self):
        """Testa a função plot_technical_indicators."""
        # Testar com indicadores válidos
        indicators = ["SMA_50", "RSI"]
        fig = plotting.plot_technical_indicators(self.test_df, indicators)
        self.assertIsInstance(fig, go.Figure, "Deveria retornar um objeto Figure do Plotly")
        self.assertEqual(len(fig.data), len(indicators), f"Deveria ter {len(indicators)} traces")
        
        # Testar com indicador inexistente
        indicators = ["SMA_50", "Indicador_Inexistente"]
        fig = plotting.plot_technical_indicators(self.test_df, indicators)
        self.assertIsInstance(fig, go.Figure, "Deveria retornar um objeto Figure do Plotly")
        self.assertEqual(len(fig.data), 1, "Deveria ter apenas 1 trace (indicador existente)")
        
        # Testar com lista vazia de indicadores
        fig = plotting.plot_technical_indicators(self.test_df, [])
        self.assertIsInstance(fig, go.Figure, "Deveria retornar um objeto Figure do Plotly mesmo com lista vazia")
        
        # Testar com DataFrame None
        fig = plotting.plot_technical_indicators(None, ["SMA_50"])
        self.assertIsInstance(fig, go.Figure, "Deveria retornar um objeto Figure do Plotly mesmo com DataFrame None")
    
    def test_plot_forecast_vs_actual(self):
        """Testa a função plot_forecast_vs_actual."""
        # Testar com dados válidos
        fig = plotting.plot_forecast_vs_actual(self.test_df, self.forecast_values, self.forecast_dates, "BTC-USD", "LSTM")
        self.assertIsInstance(fig, go.Figure, "Deveria retornar um objeto Figure do Plotly")
        self.assertEqual(len(fig.data), 2, "Deveria ter 2 traces (histórico e previsão)")
        
        # Testar com valores de previsão None
        fig = plotting.plot_forecast_vs_actual(self.test_df, None, self.forecast_dates, "BTC-USD", "LSTM")
        self.assertIsInstance(fig, go.Figure, "Deveria retornar um objeto Figure do Plotly mesmo com previsão None")
        
        # Testar com DataFrame histórico None
        fig = plotting.plot_forecast_vs_actual(None, self.forecast_values, self.forecast_dates, "BTC-USD", "LSTM")
        self.assertIsInstance(fig, go.Figure, "Deveria retornar um objeto Figure do Plotly mesmo com histórico None")
    
    def test_plot_strategy_performance(self):
        """Testa a função plot_strategy_performance."""
        # Testar com dados válidos
        fig = plotting.plot_strategy_performance(self.returns_dict, "Comparação de Estratégias")
        self.assertIsInstance(fig, go.Figure, "Deveria retornar um objeto Figure do Plotly")
        self.assertEqual(len(fig.data), len(self.returns_dict), f"Deveria ter {len(self.returns_dict)} traces")
        
        # Testar com dicionário vazio
        fig = plotting.plot_strategy_performance({}, "Comparação de Estratégias")
        self.assertIsInstance(fig, go.Figure, "Deveria retornar um objeto Figure do Plotly mesmo com dicionário vazio")
        
        # Testar com valores None no dicionário
        returns_dict_with_none = {
            "Long-Only (Forecast)": self.returns_dict["Long-Only (Forecast)"],
            "Buy-and-Hold": None
        }
        fig = plotting.plot_strategy_performance(returns_dict_with_none, "Comparação de Estratégias")
        self.assertIsInstance(fig, go.Figure, "Deveria retornar um objeto Figure do Plotly mesmo com valores None")
        self.assertEqual(len(fig.data), 1, "Deveria ter apenas 1 trace (série válida)")

if __name__ == '__main__':
    unittest.main()

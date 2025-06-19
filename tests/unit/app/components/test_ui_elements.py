# -*- coding: utf-8 -*-
"""Testes unitários para o módulo ui_elements.py."""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Adicionar diretório raiz ao path
test_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(test_dir, "..", "..", "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Importar o módulo a ser testado
# Precisamos mockar o streamlit para testes
with patch.dict('sys.modules', {'streamlit': MagicMock()}):
    import streamlit as st
    from src.app.components import ui_elements

class TestUIElements(unittest.TestCase):
    """Testes para o módulo ui_elements.py."""
    
    def setUp(self):
        """Configuração inicial para os testes."""
        # Criar dados de teste
        self.test_dates = pd.date_range(start="2023-01-01", periods=100)
        self.test_df = pd.DataFrame({
            "Adj Close": 100 + np.random.randn(100).cumsum(),
            "Volume": np.random.randint(10000, 50000, 100),
            "SMA_50": (100 + np.random.randn(100).cumsum()) * 0.98,
            "RSI": np.random.uniform(30, 70, 100),
            "MACD": np.random.randn(100)
        }, index=self.test_dates)
        
        # Criar dados de previsão
        self.forecast_dates = pd.date_range(start=self.test_dates[-1] + pd.Timedelta(days=1), periods=14)
        self.forecast_values = self.test_df["Adj Close"].iloc[-1] * (1 + np.random.randn(14)*0.02).cumsum()
        
        # Resetar mocks do streamlit
        st.reset_mock()
    
    def test_create_asset_selector(self):
        """Testa a função create_asset_selector."""
        # Configurar mock para st.selectbox
        st.selectbox = MagicMock(return_value="Bitcoin (BTC-USD)")
        
        # Testar com opção padrão
        asset_name, asset_code = ui_elements.create_asset_selector()
        self.assertEqual(asset_name, "Bitcoin (BTC-USD)", "Nome do ativo deveria ser 'Bitcoin (BTC-USD)'")
        self.assertEqual(asset_code, "btc", "Código do ativo deveria ser 'btc'")
        
        # Verificar se st.selectbox foi chamado corretamente
        st.selectbox.assert_called_once()
        
        # Testar com opção padrão diferente
        st.reset_mock()
        st.selectbox = MagicMock(return_value="Apple (AAPL)")
        asset_name, asset_code = ui_elements.create_asset_selector(default_option="Apple (AAPL)")
        self.assertEqual(asset_name, "Apple (AAPL)", "Nome do ativo deveria ser 'Apple (AAPL)'")
        self.assertEqual(asset_code, "aapl", "Código do ativo deveria ser 'aapl'")
    
    def test_create_model_selector(self):
        """Testa a função create_model_selector."""
        # Configurar mock para st.selectbox
        st.selectbox = MagicMock(return_value="LSTM")
        
        # Testar com opção padrão
        model_name, model_code = ui_elements.create_model_selector()
        self.assertEqual(model_name, "LSTM", "Nome do modelo deveria ser 'LSTM'")
        self.assertEqual(model_code, "lstm", "Código do modelo deveria ser 'lstm'")
        
        # Verificar se st.selectbox foi chamado corretamente
        st.selectbox.assert_called_once()
        
        # Testar com opção padrão diferente
        st.reset_mock()
        st.selectbox = MagicMock(return_value="GRU")
        model_name, model_code = ui_elements.create_model_selector(default_option="GRU")
        self.assertEqual(model_name, "GRU", "Nome do modelo deveria ser 'GRU'")
        self.assertEqual(model_code, "gru", "Código do modelo deveria ser 'gru'")
    
    def test_create_date_range_selector(self):
        """Testa a função create_date_range_selector."""
        # Configurar mocks para st.columns, st.date_input
        start_date = self.test_dates[-90].date()
        end_date = self.test_dates[-1].date()
        
        # Criar mocks para as colunas
        col1_mock = MagicMock()
        col2_mock = MagicMock()
        col1_mock.date_input.return_value = start_date
        col2_mock.date_input.return_value = end_date
        
        # Configurar o mock de columns para retornar as colunas mockadas
        st.columns = MagicMock(return_value=[col1_mock, col2_mock])
        
        # Testar com DataFrame válido
        result_start, result_end = ui_elements.create_date_range_selector(self.test_df)
        
        # Verificar resultados
        self.assertEqual(result_start, start_date, "Data de início deveria ser a configurada no mock")
        self.assertEqual(result_end, end_date, "Data de fim deveria ser a configurada no mock")
        
        # Verificar se st.columns foi chamado corretamente
        st.columns.assert_called_once_with(2)
        
        # Testar com DataFrame None
        st.reset_mock()
        st.warning = MagicMock()
        result = ui_elements.create_date_range_selector(None)
        self.assertEqual(result, (None, None), "Resultado deveria ser (None, None) para DataFrame None")
        st.warning.assert_called_once()
    
    def test_create_indicator_selector(self):
        """Testa a função create_indicator_selector."""
        # Configurar mock para st.multiselect
        st.multiselect = MagicMock(return_value=["SMA_50", "RSI"])
        
        # Testar com DataFrame válido
        indicators = ui_elements.create_indicator_selector(self.test_df)
        self.assertEqual(indicators, ["SMA_50", "RSI"], "Indicadores selecionados deveriam ser 'SMA_50' e 'RSI'")
        
        # Verificar se st.multiselect foi chamado corretamente
        st.multiselect.assert_called_once()
        
        # Testar com DataFrame None
        st.reset_mock()
        st.warning = MagicMock()
        indicators = ui_elements.create_indicator_selector(None)
        self.assertEqual(indicators, [], "Resultado deveria ser lista vazia para DataFrame None")
        st.warning.assert_called_once()
    
    def test_create_forecast_button(self):
        """Testa a função create_forecast_button."""
        # Configurar mock para st.button
        st.button = MagicMock(return_value=True)
        
        # Testar com valores válidos
        result = ui_elements.create_forecast_button("BTC-USD", "LSTM")
        self.assertTrue(result, "Resultado deveria ser True (botão clicado)")
        
        # Verificar se st.button foi chamado corretamente
        st.button.assert_called_once()
    
    def test_display_metrics_card(self):
        """Testa a função display_metrics_card."""
        # Configurar mocks para st.subheader, st.columns
        st.subheader = MagicMock()
        
        # Criar mocks para as colunas
        col1_mock = MagicMock()
        col2_mock = MagicMock()
        col3_mock = MagicMock()
        
        # Configurar o mock de columns para retornar as colunas mockadas
        st.columns = MagicMock(return_value=[col1_mock, col2_mock, col3_mock])
        
        # Testar com métricas válidas
        metrics = {
            "RMSE": 123.45,
            "MAE": 98.76,
            "Sharpe Ratio": 1.23
        }
        ui_elements.display_metrics_card(metrics, "Métricas de Teste")
        
        # Verificar se st.subheader foi chamado corretamente
        st.subheader.assert_called_once_with("Métricas de Teste")
        
        # Verificar se st.columns foi chamado corretamente
        st.columns.assert_called_once_with(len(metrics))
    
    def test_display_forecast_table(self):
        """Testa a função display_forecast_table."""
        # Configurar mocks para st.subheader, st.dataframe
        st.subheader = MagicMock()
        st.dataframe = MagicMock()
        
        # Testar com dados válidos
        ui_elements.display_forecast_table(self.forecast_values, self.forecast_dates)
        
        # Verificar se st.subheader foi chamado corretamente
        st.subheader.assert_called_once_with("Valores Previstos")
        
        # Verificar se st.dataframe foi chamado
        st.dataframe.assert_called_once()
        
        # Testar com dados None
        st.reset_mock()
        st.warning = MagicMock()
        ui_elements.display_forecast_table(None, self.forecast_dates)
        st.warning.assert_called_once()
        st.dataframe.assert_not_called()
    
    def test_create_sidebar_info(self):
        """Testa a função create_sidebar_info."""
        # Configurar mocks para st.sidebar
        st.sidebar.markdown = MagicMock()
        st.sidebar.subheader = MagicMock()
        
        # Testar com valores válidos
        ui_elements.create_sidebar_info("btc", "lstm")
        
        # Verificar se st.sidebar.markdown foi chamado várias vezes
        self.assertTrue(st.sidebar.markdown.call_count >= 3, 
                        "st.sidebar.markdown deveria ser chamado pelo menos 3 vezes")
        
        # Verificar se st.sidebar.subheader foi chamado
        st.sidebar.subheader.assert_called_once_with("Informações")

if __name__ == '__main__':
    unittest.main()

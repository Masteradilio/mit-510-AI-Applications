#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testes Unitários para Métricas de Avaliação Financeira

Este módulo contém testes para validar o funcionamento correto das métricas
de avaliação implementadas no módulo evaluation_metrics.py.

Autor: Sistema de IA
Data: 2024
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Adiciona o diretório src ao path para importar os módulos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from modeling.evaluation_metrics import FinancialMetrics, calculate_portfolio_metrics


class TestFinancialMetrics:
    """Testes para a classe FinancialMetrics."""
    
    def setup_method(self):
        """Configuração inicial para cada teste."""
        self.calculator = FinancialMetrics(risk_free_rate=0.02)
        
        # Dados de teste
        np.random.seed(42)
        self.n_days = 100
        self.initial_price = 100.0
        
        # Retornos simulados
        self.returns = np.random.normal(0.001, 0.02, self.n_days)
        self.prices = self.initial_price * np.cumprod(1 + self.returns)
        
        # Previsões simuladas
        self.predicted_returns = self.returns + np.random.normal(0, 0.005, self.n_days)
        self.predicted_prices = self.initial_price * np.cumprod(1 + self.predicted_returns)
        
        # Benchmark
        self.benchmark_returns = np.random.normal(0.0005, 0.015, self.n_days)
    
    def test_calculate_returns(self):
        """Testa o cálculo de retornos."""
        prices = np.array([100, 105, 102, 108])
        expected_returns = np.array([0.05, -0.0286, 0.0588])
        
        returns = self.calculator.calculate_returns(prices)
        
        assert len(returns) == len(prices) - 1
        np.testing.assert_array_almost_equal(returns, expected_returns, decimal=3)
    
    def test_calculate_returns_pandas(self):
        """Testa o cálculo de retornos com pandas Series."""
        prices = pd.Series([100, 105, 102, 108])
        returns = self.calculator.calculate_returns(prices)
        
        assert isinstance(returns, np.ndarray)
        assert len(returns) == len(prices) - 1
    
    def test_calculate_returns_insufficient_data(self):
        """Testa erro com dados insuficientes."""
        with pytest.raises(ValueError, match="É necessário pelo menos 2 preços"):
            self.calculator.calculate_returns(np.array([100]))
    
    def test_sharpe_ratio(self):
        """Testa o cálculo do Sharpe Ratio."""
        # Retornos positivos constantes
        returns = np.full(252, 0.001)  # 0.1% diário
        sharpe = self.calculator.sharpe_ratio(returns)
        
        # Sharpe deve ser positivo para retornos consistentemente positivos
        assert sharpe > 0
        assert isinstance(sharpe, float)
    
    def test_sharpe_ratio_zero_volatility(self):
        """Testa Sharpe Ratio com volatilidade zero."""
        returns = np.zeros(100)
        sharpe = self.calculator.sharpe_ratio(returns)
        
        assert sharpe == 0.0
    
    def test_sharpe_ratio_empty_array(self):
        """Testa Sharpe Ratio com array vazio."""
        returns = np.array([])
        sharpe = self.calculator.sharpe_ratio(returns)
        
        assert sharpe == 0.0
    
    def test_maximum_drawdown(self):
        """Testa o cálculo do Maximum Drawdown."""
        # Preços que sobem e depois caem
        prices = np.array([100, 110, 120, 90, 95, 85])
        
        max_dd, peak_idx, valley_idx = self.calculator.maximum_drawdown(prices)
        
        assert max_dd > 0  # Deve haver drawdown
        assert peak_idx <= valley_idx  # Pico deve vir antes do vale
        assert isinstance(max_dd, float)
        assert isinstance(peak_idx, (int, np.integer))
        assert isinstance(valley_idx, (int, np.integer))
    
    def test_maximum_drawdown_no_decline(self):
        """Testa Maximum Drawdown com preços sempre crescentes."""
        prices = np.array([100, 105, 110, 115, 120])
        
        max_dd, peak_idx, valley_idx = self.calculator.maximum_drawdown(prices)
        
        assert max_dd == 0.0  # Não deve haver drawdown
    
    def test_maximum_drawdown_insufficient_data(self):
        """Testa Maximum Drawdown com dados insuficientes."""
        prices = np.array([100])
        
        max_dd, peak_idx, valley_idx = self.calculator.maximum_drawdown(prices)
        
        assert max_dd == 0.0
        assert peak_idx == 0
        assert valley_idx == 0
    
    def test_information_ratio(self):
        """Testa o cálculo do Information Ratio."""
        portfolio_returns = np.random.normal(0.001, 0.02, 252)
        benchmark_returns = np.random.normal(0.0005, 0.015, 252)
        
        ir = self.calculator.information_ratio(portfolio_returns, benchmark_returns)
        
        assert isinstance(ir, float)
        # Information Ratio pode ser positivo ou negativo
    
    def test_information_ratio_mismatched_length(self):
        """Testa erro com arrays de tamanhos diferentes."""
        portfolio_returns = np.random.normal(0.001, 0.02, 100)
        benchmark_returns = np.random.normal(0.0005, 0.015, 50)
        
        with pytest.raises(ValueError, match="mesmo número de observações"):
            self.calculator.information_ratio(portfolio_returns, benchmark_returns)
    
    def test_information_ratio_zero_tracking_error(self):
        """Testa Information Ratio com tracking error zero."""
        returns = np.random.normal(0.001, 0.02, 100)
        
        ir = self.calculator.information_ratio(returns, returns)
        
        assert ir == 0.0
    
    def test_sortino_ratio(self):
        """Testa o cálculo do Sortino Ratio."""
        returns = np.random.normal(0.001, 0.02, 252)
        
        sortino = self.calculator.sortino_ratio(returns)
        
        assert isinstance(sortino, float)
    
    def test_sortino_ratio_no_downside(self):
        """Testa Sortino Ratio sem retornos negativos."""
        returns = np.abs(np.random.normal(0.001, 0.02, 100))  # Todos positivos
        
        sortino = self.calculator.sortino_ratio(returns)
        
        assert sortino == 0.0  # Sem downside deviation
    
    def test_directional_accuracy(self):
        """Testa o cálculo da acurácia direcional."""
        actual_prices = np.array([100, 105, 102, 108, 110])
        predicted_prices = np.array([100, 104, 103, 107, 112])
        
        metrics = self.calculator.directional_accuracy(actual_prices, predicted_prices)
        
        assert 'directional_accuracy' in metrics
        assert 'up_accuracy' in metrics
        assert 'down_accuracy' in metrics
        assert 'total_predictions' in metrics
        
        assert 0 <= metrics['directional_accuracy'] <= 1
        assert 0 <= metrics['up_accuracy'] <= 1
        assert 0 <= metrics['down_accuracy'] <= 1
        assert metrics['total_predictions'] == len(actual_prices) - 1
    
    def test_directional_accuracy_perfect_prediction(self):
        """Testa acurácia direcional com previsão perfeita."""
        prices = np.array([100, 105, 102, 108, 110])
        
        metrics = self.calculator.directional_accuracy(prices, prices)
        
        assert metrics['directional_accuracy'] == 1.0
    
    def test_directional_accuracy_insufficient_data(self):
        """Testa acurácia direcional com dados insuficientes."""
        actual_prices = np.array([100])
        predicted_prices = np.array([100])
        
        metrics = self.calculator.directional_accuracy(actual_prices, predicted_prices)
        
        assert metrics['directional_accuracy'] == 0.0
        assert metrics['total_predictions'] == 0
    
    def test_directional_accuracy_mismatched_length(self):
        """Testa erro com arrays de tamanhos diferentes."""
        actual_prices = np.array([100, 105, 102])
        predicted_prices = np.array([100, 104])
        
        with pytest.raises(ValueError, match="mesmo tamanho"):
            self.calculator.directional_accuracy(actual_prices, predicted_prices)
    
    def test_calmar_ratio(self):
        """Testa o cálculo do Calmar Ratio."""
        returns = np.random.normal(0.001, 0.02, 252)
        
        calmar = self.calculator.calmar_ratio(returns)
        
        assert isinstance(calmar, float)
    
    def test_calmar_ratio_no_drawdown(self):
        """Testa Calmar Ratio sem drawdown."""
        returns = np.abs(np.random.normal(0.001, 0.02, 100))  # Sempre positivos
        
        calmar = self.calculator.calmar_ratio(returns)
        
        assert calmar == 0.0  # Sem drawdown
    
    def test_calculate_all_metrics(self):
        """Testa o cálculo de todas as métricas."""
        metrics = self.calculator.calculate_all_metrics(
            self.prices, 
            self.predicted_prices
        )
        
        expected_keys = [
            'sharpe_ratio_actual', 'sharpe_ratio_predicted',
            'sortino_ratio_actual', 'sortino_ratio_predicted',
            'calmar_ratio_actual', 'calmar_ratio_predicted',
            'max_drawdown_actual', 'max_drawdown_predicted',
            'directional_accuracy', 'up_accuracy', 'down_accuracy',
            'total_predictions', 'up_predictions', 'down_predictions'
        ]
        
        for key in expected_keys:
            assert key in metrics
    
    def test_calculate_all_metrics_with_benchmark(self):
        """Testa o cálculo de todas as métricas com benchmark."""
        benchmark_prices = self.initial_price * np.cumprod(1 + self.benchmark_returns)
        
        metrics = self.calculator.calculate_all_metrics(
            self.prices, 
            self.predicted_prices,
            benchmark_prices
        )
        
        assert 'information_ratio_actual' in metrics
        assert 'information_ratio_predicted' in metrics
    
    def test_pandas_series_input(self):
        """Testa entrada com pandas Series."""
        prices_series = pd.Series(self.prices)
        predicted_series = pd.Series(self.predicted_prices)
        
        metrics = self.calculator.calculate_all_metrics(
            prices_series, 
            predicted_series
        )
        
        assert len(metrics) > 0
        assert 'directional_accuracy' in metrics


class TestCalculatePortfolioMetrics:
    """Testes para a função calculate_portfolio_metrics."""
    
    def setup_method(self):
        """Configuração inicial para cada teste."""
        np.random.seed(42)
        
        # Simula dados para múltiplos ativos
        self.n_days = 100
        self.initial_price = 100.0
        
        self.prices_dict = {}
        self.predictions_dict = {}
        
        for asset in ['BTC', 'AAPL']:
            returns = np.random.normal(0.001, 0.02, self.n_days)
            prices = self.initial_price * np.cumprod(1 + returns)
            
            predicted_returns = returns + np.random.normal(0, 0.005, self.n_days)
            predicted_prices = self.initial_price * np.cumprod(1 + predicted_returns)
            
            self.prices_dict[asset] = prices
            self.predictions_dict[asset] = predicted_prices
    
    def test_calculate_portfolio_metrics(self):
        """Testa o cálculo de métricas para múltiplos ativos."""
        results = calculate_portfolio_metrics(
            self.prices_dict, 
            self.predictions_dict
        )
        
        assert 'BTC' in results
        assert 'AAPL' in results
        
        for asset in ['BTC', 'AAPL']:
            assert 'directional_accuracy' in results[asset]
            assert 'sharpe_ratio_actual' in results[asset]
    
    def test_calculate_portfolio_metrics_missing_predictions(self):
        """Testa com previsões faltantes para alguns ativos."""
        predictions_dict = {'BTC': self.predictions_dict['BTC']}  # Só BTC
        
        results = calculate_portfolio_metrics(
            self.prices_dict, 
            predictions_dict
        )
        
        assert 'BTC' in results
        assert 'AAPL' not in results


if __name__ == "__main__":
    # Executa os testes
    pytest.main([__file__, "-v"])
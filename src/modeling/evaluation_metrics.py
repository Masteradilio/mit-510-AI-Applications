#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo de Métricas de Avaliação Específicas para Modelos Financeiros

Este módulo implementa métricas de avaliação específicas para modelos de previsão
financeira, incluindo métricas de risco-retorno e acurácia direcional.

Autor: Sistema de IA
Data: 2024
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional
import warnings


class FinancialMetrics:
    """
    Classe para cálculo de métricas de avaliação específicas para modelos financeiros.
    
    Esta classe implementa métricas como Sharpe Ratio, Maximum Drawdown, Information Ratio,
    Sortino Ratio e métricas de acurácia direcional.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Inicializa a classe com taxa livre de risco.
        
        Args:
            risk_free_rate (float): Taxa livre de risco anual (padrão: 2%)
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_returns(self, prices: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        Calcula os retornos percentuais a partir dos preços.
        
        Args:
            prices: Array ou Series com os preços
            
        Returns:
            np.ndarray: Array com os retornos percentuais
        """
        if isinstance(prices, pd.Series):
            prices = prices.values
        
        if len(prices) < 2:
            raise ValueError("É necessário pelo menos 2 preços para calcular retornos")
        
        returns = np.diff(prices) / prices[:-1]
        return returns
    
    def sharpe_ratio(self, 
                    returns: Union[np.ndarray, pd.Series], 
                    risk_free_rate: Optional[float] = None) -> float:
        """
        Calcula o Sharpe Ratio.
        
        O Sharpe Ratio mede o retorno ajustado ao risco, calculado como:
        (Retorno Médio - Taxa Livre de Risco) / Desvio Padrão dos Retornos
        
        Args:
            returns: Array ou Series com os retornos
            risk_free_rate: Taxa livre de risco (se None, usa a da classe)
            
        Returns:
            float: Sharpe Ratio
        """
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        if len(returns) == 0:
            return 0.0
        
        rf_rate = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        
        # Converte taxa anual para período dos retornos (assumindo retornos diários)
        daily_rf_rate = rf_rate / 252
        
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        if std_return == 0:
            return 0.0
        
        sharpe = (mean_return - daily_rf_rate) / std_return
        
        # Anualiza o Sharpe Ratio
        return sharpe * np.sqrt(252)
    
    def maximum_drawdown(self, prices: Union[np.ndarray, pd.Series]) -> Tuple[float, int, int]:
        """
        Calcula o Maximum Drawdown.
        
        O Maximum Drawdown é a maior queda percentual do pico ao vale
        durante um período específico.
        
        Args:
            prices: Array ou Series com os preços
            
        Returns:
            Tuple[float, int, int]: (max_drawdown, início_do_drawdown, fim_do_drawdown)
        """
        if isinstance(prices, pd.Series):
            prices = prices.values
        
        if len(prices) < 2:
            return 0.0, 0, 0
        
        # Calcula os picos cumulativos
        peak = np.maximum.accumulate(prices)
        
        # Calcula o drawdown
        drawdown = (prices - peak) / peak
        
        # Encontra o maximum drawdown
        max_dd_idx = np.argmin(drawdown)
        max_drawdown = drawdown[max_dd_idx]
        
        # Encontra o início do drawdown (último pico antes do vale)
        peak_idx = 0
        for i in range(max_dd_idx, -1, -1):
            if prices[i] == peak[i]:
                peak_idx = i
                break
        
        return abs(max_drawdown), peak_idx, max_dd_idx
    
    def information_ratio(self, 
                         portfolio_returns: Union[np.ndarray, pd.Series],
                         benchmark_returns: Union[np.ndarray, pd.Series]) -> float:
        """
        Calcula o Information Ratio.
        
        O Information Ratio mede o retorno ativo ajustado ao risco ativo:
        (Retorno do Portfolio - Retorno do Benchmark) / Tracking Error
        
        Args:
            portfolio_returns: Retornos do portfolio
            benchmark_returns: Retornos do benchmark
            
        Returns:
            float: Information Ratio
        """
        if isinstance(portfolio_returns, pd.Series):
            portfolio_returns = portfolio_returns.values
        if isinstance(benchmark_returns, pd.Series):
            benchmark_returns = benchmark_returns.values
        
        if len(portfolio_returns) != len(benchmark_returns):
            raise ValueError("Portfolio e benchmark devem ter o mesmo número de observações")
        
        if len(portfolio_returns) == 0:
            return 0.0
        
        # Calcula retornos ativos
        active_returns = portfolio_returns - benchmark_returns
        
        mean_active_return = np.mean(active_returns)
        tracking_error = np.std(active_returns, ddof=1)
        
        if tracking_error == 0:
            return 0.0
        
        # Anualiza o Information Ratio
        return (mean_active_return / tracking_error) * np.sqrt(252)
    
    def sortino_ratio(self, 
                     returns: Union[np.ndarray, pd.Series],
                     target_return: float = 0.0,
                     risk_free_rate: Optional[float] = None) -> float:
        """
        Calcula o Sortino Ratio.
        
        O Sortino Ratio é similar ao Sharpe Ratio, mas considera apenas
        a volatilidade dos retornos negativos (downside deviation).
        
        Args:
            returns: Array ou Series com os retornos
            target_return: Retorno alvo (padrão: 0%)
            risk_free_rate: Taxa livre de risco (se None, usa a da classe)
            
        Returns:
            float: Sortino Ratio
        """
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        if len(returns) == 0:
            return 0.0
        
        rf_rate = risk_free_rate if risk_free_rate is not None else self.risk_free_rate
        
        # Converte taxa anual para período dos retornos
        daily_rf_rate = rf_rate / 252
        daily_target = target_return / 252
        
        mean_return = np.mean(returns)
        
        # Calcula downside deviation
        downside_returns = returns[returns < daily_target]
        if len(downside_returns) == 0:
            downside_deviation = 0.0
        else:
            downside_deviation = np.sqrt(np.mean((downside_returns - daily_target) ** 2))
        
        if downside_deviation == 0:
            return 0.0
        
        sortino = (mean_return - daily_rf_rate) / downside_deviation
        
        # Anualiza o Sortino Ratio
        return sortino * np.sqrt(252)
    
    def directional_accuracy(self, 
                           actual_prices: Union[np.ndarray, pd.Series],
                           predicted_prices: Union[np.ndarray, pd.Series]) -> dict:
        """
        Calcula métricas de acurácia direcional.
        
        Args:
            actual_prices: Preços reais
            predicted_prices: Preços previstos
            
        Returns:
            dict: Dicionário com métricas de acurácia direcional
        """
        if isinstance(actual_prices, pd.Series):
            actual_prices = actual_prices.values
        if isinstance(predicted_prices, pd.Series):
            predicted_prices = predicted_prices.values
        
        if len(actual_prices) != len(predicted_prices):
            raise ValueError("Arrays de preços devem ter o mesmo tamanho")
        
        if len(actual_prices) < 2:
            return {
                'directional_accuracy': 0.0,
                'up_accuracy': 0.0,
                'down_accuracy': 0.0,
                'total_predictions': 0
            }
        
        # Calcula direções (subida/descida)
        actual_directions = np.diff(actual_prices) > 0
        predicted_directions = np.diff(predicted_prices) > 0
        
        # Acurácia direcional geral
        correct_directions = actual_directions == predicted_directions
        directional_accuracy = np.mean(correct_directions)
        
        # Acurácia para movimentos de subida
        up_mask = actual_directions == True
        if np.sum(up_mask) > 0:
            up_accuracy = np.mean(correct_directions[up_mask])
        else:
            up_accuracy = 0.0
        
        # Acurácia para movimentos de descida
        down_mask = actual_directions == False
        if np.sum(down_mask) > 0:
            down_accuracy = np.mean(correct_directions[down_mask])
        else:
            down_accuracy = 0.0
        
        return {
            'directional_accuracy': float(directional_accuracy),
            'up_accuracy': float(up_accuracy),
            'down_accuracy': float(down_accuracy),
            'total_predictions': len(correct_directions),
            'up_predictions': int(np.sum(up_mask)),
            'down_predictions': int(np.sum(down_mask))
        }
    
    def calmar_ratio(self, returns: Union[np.ndarray, pd.Series]) -> float:
        """
        Calcula o Calmar Ratio.
        
        O Calmar Ratio é o retorno anualizado dividido pelo Maximum Drawdown.
        
        Args:
            returns: Array ou Series com os retornos
            
        Returns:
            float: Calmar Ratio
        """
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        if len(returns) == 0:
            return 0.0
        
        # Calcula retorno anualizado
        annual_return = np.mean(returns) * 252
        
        # Calcula preços cumulativos para o Maximum Drawdown
        cumulative_returns = np.cumprod(1 + returns)
        max_dd, _, _ = self.maximum_drawdown(cumulative_returns)
        
        if max_dd == 0:
            return 0.0
        
        return annual_return / max_dd
    
    def calculate_all_metrics(self, 
                            actual_prices: Union[np.ndarray, pd.Series],
                            predicted_prices: Union[np.ndarray, pd.Series],
                            benchmark_prices: Optional[Union[np.ndarray, pd.Series]] = None) -> dict:
        """
        Calcula todas as métricas de avaliação.
        
        Args:
            actual_prices: Preços reais
            predicted_prices: Preços previstos
            benchmark_prices: Preços do benchmark (opcional)
            
        Returns:
            dict: Dicionário com todas as métricas calculadas
        """
        try:
            # Calcula retornos
            actual_returns = self.calculate_returns(actual_prices)
            predicted_returns = self.calculate_returns(predicted_prices)
            
            # Métricas básicas
            metrics = {
                'sharpe_ratio_actual': self.sharpe_ratio(actual_returns),
                'sharpe_ratio_predicted': self.sharpe_ratio(predicted_returns),
                'sortino_ratio_actual': self.sortino_ratio(actual_returns),
                'sortino_ratio_predicted': self.sortino_ratio(predicted_returns),
                'calmar_ratio_actual': self.calmar_ratio(actual_returns),
                'calmar_ratio_predicted': self.calmar_ratio(predicted_returns)
            }
            
            # Maximum Drawdown
            max_dd_actual, _, _ = self.maximum_drawdown(actual_prices)
            max_dd_predicted, _, _ = self.maximum_drawdown(predicted_prices)
            
            metrics.update({
                'max_drawdown_actual': max_dd_actual,
                'max_drawdown_predicted': max_dd_predicted
            })
            
            # Acurácia direcional
            directional_metrics = self.directional_accuracy(actual_prices, predicted_prices)
            metrics.update(directional_metrics)
            
            # Information Ratio (se benchmark fornecido)
            if benchmark_prices is not None:
                benchmark_returns = self.calculate_returns(benchmark_prices)
                if len(benchmark_returns) == len(actual_returns):
                    metrics['information_ratio_actual'] = self.information_ratio(
                        actual_returns, benchmark_returns
                    )
                    metrics['information_ratio_predicted'] = self.information_ratio(
                        predicted_returns, benchmark_returns
                    )
            
            return metrics
            
        except Exception as e:
            warnings.warn(f"Erro ao calcular métricas: {str(e)}")
            return {}


def calculate_portfolio_metrics(prices_dict: dict, 
                              predictions_dict: dict,
                              risk_free_rate: float = 0.02) -> dict:
    """
    Calcula métricas para múltiplos ativos.
    
    Args:
        prices_dict: Dicionário com preços reais {ativo: preços}
        predictions_dict: Dicionário com previsões {ativo: previsões}
        risk_free_rate: Taxa livre de risco
        
    Returns:
        dict: Métricas por ativo e métricas agregadas
    """
    metrics_calculator = FinancialMetrics(risk_free_rate)
    results = {}
    
    for asset in prices_dict.keys():
        if asset in predictions_dict:
            asset_metrics = metrics_calculator.calculate_all_metrics(
                prices_dict[asset],
                predictions_dict[asset]
            )
            results[asset] = asset_metrics
    
    return results


if __name__ == "__main__":
    # Exemplo de uso
    np.random.seed(42)
    
    # Simula dados de preços
    n_days = 252
    initial_price = 100
    returns = np.random.normal(0.001, 0.02, n_days)
    prices = initial_price * np.cumprod(1 + returns)
    
    # Simula previsões (com algum ruído)
    predicted_returns = returns + np.random.normal(0, 0.005, n_days)
    predicted_prices = initial_price * np.cumprod(1 + predicted_returns)
    
    # Calcula métricas
    calculator = FinancialMetrics()
    metrics = calculator.calculate_all_metrics(prices, predicted_prices)
    
    print("Métricas de Avaliação Financeira:")
    print("=" * 40)
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")
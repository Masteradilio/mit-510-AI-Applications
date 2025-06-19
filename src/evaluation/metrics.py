"""Módulo de métricas de avaliação para modelos de previsão financeira.

Este módulo implementa as métricas específicas mencionadas no documento de qualidade:
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- Sharpe Ratio
- Outras métricas financeiras relevantes
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Union, Tuple, Dict, Any

# Importar a nova classe de métricas financeiras
try:
    from ..modeling.evaluation_metrics import FinancialMetrics, calculate_portfolio_metrics
except ImportError:
    # Fallback se o módulo não estiver disponível
    FinancialMetrics = None
    calculate_portfolio_metrics = None


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula o Root Mean Square Error (RMSE).
    
    Args:
        y_true (np.ndarray): Valores reais.
        y_pred (np.ndarray): Valores preditos.
        
    Returns:
        float: RMSE.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula o Mean Absolute Error (MAE).
    
    Args:
        y_true (np.ndarray): Valores reais.
        y_pred (np.ndarray): Valores preditos.
        
    Returns:
        float: MAE.
    """
    return mean_absolute_error(y_true, y_pred)


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Calcula o Mean Absolute Percentage Error (MAPE).
    
    Args:
        y_true (np.ndarray): Valores reais.
        y_pred (np.ndarray): Valores preditos.
        epsilon (float): Valor pequeno para evitar divisão por zero.
        
    Returns:
        float: MAPE em porcentagem.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Evitar divisão por zero
    y_true_safe = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
    
    return np.mean(np.abs((y_true - y_pred) / y_true_safe)) * 100


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02, 
                          periods_per_year: int = 252) -> float:
    """Calcula o Sharpe Ratio.
    
    Args:
        returns (np.ndarray): Retornos do ativo/estratégia.
        risk_free_rate (float): Taxa livre de risco anual.
        periods_per_year (int): Número de períodos por ano (252 para dias úteis).
        
    Returns:
        float: Sharpe Ratio.
    """
    if len(returns) == 0:
        return 0.0
    
    # Converter taxa livre de risco para o período
    risk_free_period = risk_free_rate / periods_per_year
    
    # Calcular excesso de retorno
    excess_returns = returns - risk_free_period
    
    # Calcular Sharpe Ratio
    if np.std(excess_returns) == 0:
        return 0.0
    
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods_per_year)


def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.02,
                           periods_per_year: int = 252) -> float:
    """Calcula o Sortino Ratio (similar ao Sharpe, mas usa apenas downside deviation).
    
    Args:
        returns (np.ndarray): Retornos do ativo/estratégia.
        risk_free_rate (float): Taxa livre de risco anual.
        periods_per_year (int): Número de períodos por ano.
        
    Returns:
        float: Sortino Ratio.
    """
    if len(returns) == 0:
        return 0.0
    
    # Converter taxa livre de risco para o período
    risk_free_period = risk_free_rate / periods_per_year
    
    # Calcular excesso de retorno
    excess_returns = returns - risk_free_period
    
    # Calcular downside deviation (apenas retornos negativos)
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf if np.mean(excess_returns) > 0 else 0.0
    
    downside_deviation = np.std(downside_returns)
    
    if downside_deviation == 0:
        return 0.0
    
    return np.mean(excess_returns) / downside_deviation * np.sqrt(periods_per_year)


def calculate_maximum_drawdown(prices: np.ndarray) -> Tuple[float, int, int]:
    """Calcula o Maximum Drawdown.
    
    Args:
        prices (np.ndarray): Série de preços ou valores de portfólio.
        
    Returns:
        Tuple[float, int, int]: (max_drawdown, start_idx, end_idx)
    """
    if len(prices) == 0:
        return 0.0, 0, 0
    
    # Calcular running maximum
    running_max = np.maximum.accumulate(prices)
    
    # Calcular drawdown
    drawdown = (prices - running_max) / running_max
    
    # Encontrar maximum drawdown
    max_dd_idx = np.argmin(drawdown)
    max_drawdown = drawdown[max_dd_idx]
    
    # Encontrar início do drawdown
    start_idx = np.argmax(running_max[:max_dd_idx + 1])
    
    return abs(max_drawdown), start_idx, max_dd_idx


def calculate_calmar_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Calcula o Calmar Ratio (retorno anualizado / maximum drawdown).
    
    Args:
        returns (np.ndarray): Retornos do ativo/estratégia.
        periods_per_year (int): Número de períodos por ano.
        
    Returns:
        float: Calmar Ratio.
    """
    if len(returns) == 0:
        return 0.0
    
    # Calcular retorno anualizado
    annualized_return = np.mean(returns) * periods_per_year
    
    # Calcular preços cumulativos para maximum drawdown
    cumulative_returns = np.cumprod(1 + returns)
    max_dd, _, _ = calculate_maximum_drawdown(cumulative_returns)
    
    if max_dd == 0:
        return np.inf if annualized_return > 0 else 0.0
    
    return annualized_return / max_dd


def calculate_information_ratio(portfolio_returns: np.ndarray, 
                               benchmark_returns: np.ndarray) -> float:
    """Calcula o Information Ratio.
    
    Args:
        portfolio_returns (np.ndarray): Retornos do portfólio.
        benchmark_returns (np.ndarray): Retornos do benchmark.
        
    Returns:
        float: Information Ratio.
    """
    if len(portfolio_returns) != len(benchmark_returns):
        raise ValueError("Portfolio e benchmark devem ter o mesmo tamanho")
    
    # Calcular excess returns
    excess_returns = portfolio_returns - benchmark_returns
    
    # Calcular tracking error
    tracking_error = np.std(excess_returns)
    
    if tracking_error == 0:
        return 0.0
    
    return np.mean(excess_returns) / tracking_error


def calculate_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcula a precisão direcional (acerto na direção do movimento).
    
    Args:
        y_true (np.ndarray): Valores reais.
        y_pred (np.ndarray): Valores preditos.
        
    Returns:
        float: Precisão direcional (0-1).
    """
    if len(y_true) <= 1 or len(y_pred) <= 1:
        return 0.0
    
    # Calcular direções (subida/descida)
    true_direction = np.diff(y_true) > 0
    pred_direction = np.diff(y_pred) > 0
    
    # Calcular precisão
    return np.mean(true_direction == pred_direction)


def calculate_hit_ratio(y_true: np.ndarray, y_pred: np.ndarray, 
                       threshold: float = 0.01) -> float:
    """Calcula o Hit Ratio (proporção de previsões dentro de um threshold).
    
    Args:
        y_true (np.ndarray): Valores reais.
        y_pred (np.ndarray): Valores preditos.
        threshold (float): Threshold de erro aceitável (em %).
        
    Returns:
        float: Hit Ratio (0-1).
    """
    percentage_error = np.abs((y_true - y_pred) / y_true)
    return np.mean(percentage_error <= threshold)


def calculate_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                   returns: np.ndarray = None,
                                   benchmark_returns: np.ndarray = None,
                                   risk_free_rate: float = 0.02) -> dict:
    """Calcula um conjunto abrangente de métricas de avaliação.
    
    Args:
        y_true (np.ndarray): Valores reais.
        y_pred (np.ndarray): Valores preditos.
        returns (np.ndarray, optional): Retornos para métricas financeiras.
        benchmark_returns (np.ndarray, optional): Retornos do benchmark.
        risk_free_rate (float): Taxa livre de risco.
        
    Returns:
        dict: Dicionário com todas as métricas calculadas.
    """
    metrics = {}
    
    # Métricas básicas de erro
    metrics['RMSE'] = calculate_rmse(y_true, y_pred)
    metrics['MAE'] = calculate_mae(y_true, y_pred)
    metrics['MAPE'] = calculate_mape(y_true, y_pred)
    
    # Métricas direcionais
    metrics['Directional_Accuracy'] = calculate_directional_accuracy(y_true, y_pred)
    metrics['Hit_Ratio_1%'] = calculate_hit_ratio(y_true, y_pred, 0.01)
    metrics['Hit_Ratio_5%'] = calculate_hit_ratio(y_true, y_pred, 0.05)
    
    # Métricas financeiras (se retornos fornecidos)
    if returns is not None:
        metrics['Sharpe_Ratio'] = calculate_sharpe_ratio(returns, risk_free_rate)
        metrics['Sortino_Ratio'] = calculate_sortino_ratio(returns, risk_free_rate)
        metrics['Calmar_Ratio'] = calculate_calmar_ratio(returns)
        
        # Maximum Drawdown
        cumulative_returns = np.cumprod(1 + returns)
        max_dd, _, _ = calculate_maximum_drawdown(cumulative_returns)
        metrics['Maximum_Drawdown'] = max_dd
        
        # Volatilidade anualizada
        metrics['Annualized_Volatility'] = np.std(returns) * np.sqrt(252)
        
        # Retorno anualizado
        metrics['Annualized_Return'] = np.mean(returns) * 252
    
    # Information Ratio (se benchmark fornecido)
    if returns is not None and benchmark_returns is not None:
        metrics['Information_Ratio'] = calculate_information_ratio(returns, benchmark_returns)
    
    return metrics


def calculate_enhanced_metrics(actual_prices: np.ndarray, 
                              predicted_prices: np.ndarray,
                              benchmark_prices: np.ndarray = None,
                              risk_free_rate: float = 0.02) -> Dict[str, Any]:
    """Calcula métricas avançadas usando a nova classe FinancialMetrics.
    
    Args:
        actual_prices (np.ndarray): Preços reais.
        predicted_prices (np.ndarray): Preços preditos.
        benchmark_prices (np.ndarray, optional): Preços do benchmark.
        risk_free_rate (float): Taxa livre de risco.
        
    Returns:
        Dict[str, Any]: Dicionário com métricas avançadas.
    """
    if FinancialMetrics is None:
        # Fallback para métricas básicas se a classe não estiver disponível
        return calculate_comprehensive_metrics(
            actual_prices, predicted_prices, 
            risk_free_rate=risk_free_rate
        )
    
    # Usar a nova classe de métricas financeiras
    calculator = FinancialMetrics(risk_free_rate=risk_free_rate)
    
    return calculator.calculate_all_metrics(
        actual_prices, 
        predicted_prices, 
        benchmark_prices
    )


def evaluate_multiple_assets(prices_dict: Dict[str, np.ndarray],
                            predictions_dict: Dict[str, np.ndarray],
                            benchmark_dict: Dict[str, np.ndarray] = None,
                            risk_free_rate: float = 0.02) -> Dict[str, Dict[str, Any]]:
    """Avalia métricas para múltiplos ativos.
    
    Args:
        prices_dict (Dict[str, np.ndarray]): Dicionário com preços reais por ativo.
        predictions_dict (Dict[str, np.ndarray]): Dicionário com previsões por ativo.
        benchmark_dict (Dict[str, np.ndarray], optional): Dicionário com benchmarks.
        risk_free_rate (float): Taxa livre de risco.
        
    Returns:
        Dict[str, Dict[str, Any]]: Métricas por ativo.
    """
    if calculate_portfolio_metrics is not None:
        # Usar a função avançada se disponível
        return calculate_portfolio_metrics(
            prices_dict, 
            predictions_dict, 
            benchmark_dict, 
            risk_free_rate
        )
    
    # Fallback para cálculo manual
    results = {}
    
    for asset in prices_dict.keys():
        if asset in predictions_dict:
            benchmark_prices = benchmark_dict.get(asset) if benchmark_dict else None
            
            results[asset] = calculate_enhanced_metrics(
                prices_dict[asset],
                predictions_dict[asset],
                benchmark_prices,
                risk_free_rate
            )
    
    return results


if __name__ == "__main__":
    # Exemplo de uso
    np.random.seed(42)
    
    # Dados de exemplo
    y_true = np.random.randn(100).cumsum() + 100
    y_pred = y_true + np.random.randn(100) * 0.5
    returns = np.random.randn(100) * 0.02
    
    # Calcular métricas
    metrics = calculate_comprehensive_metrics(y_true, y_pred, returns)
    
    print("Métricas de Avaliação:")
    print("=" * 50)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
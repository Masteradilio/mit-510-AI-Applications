# -*- coding: utf-8 -*-
"""Módulo para simulação de estratégias de trading baseadas em previsões."""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

def generate_signals(prices, predictions, threshold=0.01):
    """Gera sinais de compra/venda baseados nas previsões.
    
    Args:
        prices (pd.Series): Série de preços históricos.
        predictions (np.ndarray): Array de previsões para cada ponto no tempo.
                                 Shape: (n_samples, forecast_horizon)
        threshold (float): Limiar de retorno para gerar sinal.
        
    Returns:
        pd.Series: Série com sinais (1=compra, 0=neutro, -1=venda).
    """
    if prices is None or predictions is None:
        print("Erro: Preços ou previsões não fornecidos.")
        return None
    
    if len(prices) != len(predictions):
        print(f"Erro: Tamanhos diferentes - preços ({len(prices)}) e previsões ({len(predictions)}).")
        return None
    
    signals = pd.Series(0, index=prices.index)
    
    for i in range(len(prices)):
        # Calcular retorno previsto médio para o horizonte de previsão
        current_price = prices.iloc[i]
        forecast_horizon = predictions.shape[1]
        
        # Média das previsões para o horizonte
        avg_prediction = np.mean(predictions[i])
        
        # Calcular retorno previsto
        predicted_return = (avg_prediction / current_price) - 1
        
        # Gerar sinal baseado no retorno previsto
        if predicted_return > threshold:
            signals.iloc[i] = 1  # Compra
        elif predicted_return < -threshold:
            signals.iloc[i] = -1  # Venda
        else:
            signals.iloc[i] = 0  # Neutro
    
    return signals

def simulate_long_only_strategy(prices, predictions, forecast_horizon, initial_capital=10000):
    """Simula uma estratégia long-only baseada nas previsões.
    
    Args:
        prices (pd.Series): Série de preços históricos.
        predictions (np.ndarray): Array de previsões para cada ponto no tempo.
        forecast_horizon (int): Horizonte de previsão em dias.
        initial_capital (float): Capital inicial para a simulação.
        
    Returns:
        pd.Series: Série com retornos diários da estratégia.
    """
    if prices is None or predictions is None:
        print("Erro: Preços ou previsões não fornecidos.")
        return None
    
    # Gerar sinais
    signals = generate_signals(prices, predictions)
    
    if signals is None:
        return None
    
    # Inicializar posições e retornos
    position = 0  # 0 = sem posição, 1 = comprado
    returns = pd.Series(0.0, index=prices.index)
    
    # Simular trading
    for i in range(1, len(prices)):
        # Verificar sinal do dia anterior
        signal = signals.iloc[i-1]
        
        # Calcular retorno diário
        daily_return = prices.iloc[i] / prices.iloc[i-1] - 1
        
        # Atualizar posição baseada no sinal
        if signal == 1 and position == 0:
            # Comprar
            position = 1
            returns.iloc[i] = daily_return
        elif signal == -1 and position == 1:
            # Vender
            position = 0
            returns.iloc[i] = 0
        elif position == 1:
            # Manter posição comprada
            returns.iloc[i] = daily_return
        else:
            # Sem posição
            returns.iloc[i] = 0
    
    return returns

def simulate_buy_and_hold(prices):
    """Simula uma estratégia buy-and-hold.
    
    Args:
        prices (pd.Series): Série de preços históricos.
        
    Returns:
        pd.Series: Série com retornos diários da estratégia.
    """
    if prices is None or len(prices) < 2:
        print("Erro: Preços não fornecidos ou insuficientes.")
        return None
    
    # Calcular retornos diários
    returns = prices.pct_change()
    
    # Substituir o primeiro valor (NaN) por zero
    returns.iloc[0] = 0
    
    return returns

def calculate_sharpe_ratio(returns, risk_free_rate=0.0, annualization_factor=252):
    """Calcula o Sharpe Ratio da estratégia.
    
    Args:
        returns (pd.Series): Série com retornos diários.
        risk_free_rate (float): Taxa livre de risco anualizada.
        annualization_factor (int): Fator de anualização (252 para dias úteis).
        
    Returns:
        float: Sharpe Ratio anualizado.
    """
    if returns is None or len(returns) < 2:
        print("Erro: Retornos não fornecidos ou insuficientes.")
        return None
    
    # Converter taxa livre de risco anual para diária
    daily_risk_free = (1 + risk_free_rate) ** (1 / annualization_factor) - 1
    
    # Calcular excesso de retorno
    excess_returns = returns - daily_risk_free
    
    # Calcular média e desvio padrão dos retornos em excesso
    mean_excess_return = excess_returns.mean()
    std_excess_return = excess_returns.std()
    
    if std_excess_return == 0:
        print("Aviso: Desvio padrão zero, retornando None.")
        return None
    
    # Calcular Sharpe Ratio diário
    daily_sharpe = mean_excess_return / std_excess_return
    
    # Anualizar Sharpe Ratio
    annual_sharpe = daily_sharpe * np.sqrt(annualization_factor)
    
    return annual_sharpe

def calculate_drawdowns(returns):
    """Calcula os drawdowns da estratégia.
    
    Args:
        returns (pd.Series): Série com retornos diários.
        
    Returns:
        pd.Series: Série com drawdowns.
    """
    if returns is None or len(returns) < 2:
        print("Erro: Retornos não fornecidos ou insuficientes.")
        return None
    
    # Calcular retorno cumulativo
    cumulative_returns = (1 + returns).cumprod()
    
    # Calcular máximo acumulado até o momento
    running_max = cumulative_returns.cummax()
    
    # Calcular drawdowns
    drawdowns = (cumulative_returns / running_max) - 1
    
    return drawdowns

def calculate_performance_metrics(signals, prices, initial_capital=10000, transaction_cost=0.001):
    """Calcula métricas de performance da estratégia.
    
    Args:
        signals (pd.Series): Sinais de trading (1=compra, -1=venda, 0=neutro).
        prices (pd.Series): Série de preços históricos.
        initial_capital (float): Capital inicial para simulação.
        transaction_cost (float): Custo de transação como percentual.
        
    Returns:
        dict: Dicionário com métricas de performance.
    """
    if signals is None or prices is None or len(signals) < 2 or len(prices) < 2:
        print("Erro: Sinais ou preços não fornecidos ou insuficientes.")
        return None
    
    # Calcular retornos da estratégia
    returns = simulate_strategy_returns(prices, signals)
    
    if returns is None or len(returns) < 2:
        print("Erro: Não foi possível calcular retornos da estratégia.")
        return None
    
    # Calcular retorno cumulativo
    cumulative_return = (1 + returns).prod() - 1
    
    # Calcular retorno anualizado
    n_years = len(returns) / 252
    annualized_return = (1 + cumulative_return) ** (1 / n_years) - 1
    
    # Calcular volatilidade anualizada
    annualized_volatility = returns.std() * np.sqrt(252)
    
    # Calcular Sharpe Ratio
    sharpe_ratio = calculate_sharpe_ratio(returns)
    
    # Calcular drawdowns
    drawdowns = calculate_drawdowns(returns)
    max_drawdown = drawdowns.min()
    
    # Calcular outras métricas
    positive_days = (returns > 0).sum() / len(returns)
    
    # Contar número de trades
    num_trades = (signals.diff() != 0).sum()
    
    # Criar dicionário de métricas
    metrics = {
        "total_return": cumulative_return * 100,  # Convertido para percentual
        "annualized_return": annualized_return * 100,
        "volatility": annualized_volatility * 100,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": abs(max_drawdown) * 100,  # Valor absoluto em percentual
        "positive_days_ratio": positive_days,
        "num_trades": num_trades
    }
    
    return metrics

# ===== ESTRATÉGIAS AVANÇADAS =====

def technical_indicator_strategy(prices: pd.Series, indicators: Dict[str, pd.Series], 
                               predictions: np.ndarray = None, 
                               rsi_oversold: float = 30, rsi_overbought: float = 70,
                               bb_threshold: float = 0.02) -> pd.Series:
    """
    Estratégia baseada em sinais dos indicadores técnicos.
    
    Args:
        prices: Série de preços históricos
        indicators: Dicionário com indicadores técnicos (RSI, Bollinger Bands, etc.)
        predictions: Array de previsões (opcional)
        rsi_oversold: Nível de sobrevenda do RSI
        rsi_overbought: Nível de sobrecompra do RSI
        bb_threshold: Threshold para Bollinger Bands
        
    Returns:
        pd.Series: Sinais de trading (1=compra, -1=venda, 0=neutro)
    """
    signals = pd.Series(0, index=prices.index)
    
    # Verificar indicadores disponíveis
    has_rsi = 'RSI' in indicators
    has_bb = all(col in indicators for col in ['BB_upper', 'BB_lower', 'BB_middle'])
    has_macd = all(col in indicators for col in ['MACD', 'MACD_signal'])
    has_stoch = 'Stochastic_K' in indicators
    
    for i in range(1, len(prices)):
        buy_signals = 0
        sell_signals = 0
        total_signals = 0
        
        # Sinal RSI
        if has_rsi and not pd.isna(indicators['RSI'].iloc[i]):
            total_signals += 1
            if indicators['RSI'].iloc[i] < rsi_oversold:
                buy_signals += 1
            elif indicators['RSI'].iloc[i] > rsi_overbought:
                sell_signals += 1
        
        # Sinal Bollinger Bands
        if has_bb and not pd.isna(indicators['BB_lower'].iloc[i]):
            total_signals += 1
            price_position = (prices.iloc[i] - indicators['BB_lower'].iloc[i]) / \
                           (indicators['BB_upper'].iloc[i] - indicators['BB_lower'].iloc[i])
            
            if price_position < bb_threshold:  # Próximo da banda inferior
                buy_signals += 1
            elif price_position > (1 - bb_threshold):  # Próximo da banda superior
                sell_signals += 1
        
        # Sinal MACD
        if has_macd and not pd.isna(indicators['MACD'].iloc[i]):
            total_signals += 1
            if (indicators['MACD'].iloc[i] > indicators['MACD_signal'].iloc[i] and 
                indicators['MACD'].iloc[i-1] <= indicators['MACD_signal'].iloc[i-1]):
                buy_signals += 1
            elif (indicators['MACD'].iloc[i] < indicators['MACD_signal'].iloc[i] and 
                  indicators['MACD'].iloc[i-1] >= indicators['MACD_signal'].iloc[i-1]):
                sell_signals += 1
        
        # Sinal Stochastic
        if has_stoch and not pd.isna(indicators['Stochastic_K'].iloc[i]):
            total_signals += 1
            if indicators['Stochastic_K'].iloc[i] < 20:
                buy_signals += 1
            elif indicators['Stochastic_K'].iloc[i] > 80:
                sell_signals += 1
        
        # Decisão final baseada na maioria dos sinais
        if total_signals > 0:
            if buy_signals > sell_signals:
                signals.iloc[i] = 1
            elif sell_signals > buy_signals:
                signals.iloc[i] = -1
    
    return signals

def momentum_strategy(prices: pd.Series, predictions: np.ndarray, 
                     momentum_window: int = 20, prediction_threshold: float = 0.02) -> pd.Series:
    """
    Estratégia de momentum baseada nas previsões.
    
    Args:
        prices: Série de preços históricos
        predictions: Array de previsões
        momentum_window: Janela para cálculo do momentum
        prediction_threshold: Threshold para sinais baseados em previsões
        
    Returns:
        pd.Series: Sinais de trading
    """
    signals = pd.Series(0, index=prices.index)
    
    # Calcular momentum dos preços
    price_momentum = prices.pct_change(momentum_window)
    
    for i in range(momentum_window, len(prices)):
        # Sinal baseado no momentum dos preços
        momentum_signal = 0
        if not pd.isna(price_momentum.iloc[i]):
            if price_momentum.iloc[i] > 0.05:  # Momentum positivo forte
                momentum_signal = 1
            elif price_momentum.iloc[i] < -0.05:  # Momentum negativo forte
                momentum_signal = -1
        
        # Sinal baseado nas previsões
        prediction_signal = 0
        if i < len(predictions):
            current_price = prices.iloc[i]
            avg_prediction = np.mean(predictions[i])
            predicted_return = (avg_prediction / current_price) - 1
            
            if predicted_return > prediction_threshold:
                prediction_signal = 1
            elif predicted_return < -prediction_threshold:
                prediction_signal = -1
        
        # Combinar sinais (ambos devem concordar)
        if momentum_signal == prediction_signal and momentum_signal != 0:
            signals.iloc[i] = momentum_signal
    
    return signals

def mean_reversion_strategy(prices: pd.Series, window: int = 20, 
                          threshold: float = 2.0) -> pd.Series:
    """
    Estratégia de reversão à média.
    
    Args:
        prices: Série de preços históricos
        window: Janela para cálculo da média móvel
        threshold: Número de desvios padrão para gerar sinais
        
    Returns:
        pd.Series: Sinais de trading
    """
    signals = pd.Series(0, index=prices.index)
    
    # Calcular média móvel e desvio padrão
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    
    # Calcular z-score
    z_score = (prices - rolling_mean) / rolling_std
    
    for i in range(window, len(prices)):
        z_value = z_score.iloc[i]
        if not pd.isna(z_value):
            if z_value < -threshold:  # Preço muito abaixo da média
                signals.iloc[i] = 1  # Comprar (esperar reversão para cima)
            elif z_value > threshold:  # Preço muito acima da média
                signals.iloc[i] = -1  # Vender (esperar reversão para baixo)
    
    return signals

def risk_managed_strategy(prices: pd.Series, signals: pd.Series, 
                         stop_loss: float = 0.05, take_profit: float = 0.10,
                         max_position_size: float = 1.0) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Aplica gestão de risco com stop-loss e take-profit aos sinais.
    
    Args:
        prices: Série de preços históricos
        signals: Sinais de trading originais
        stop_loss: Percentual de stop-loss (ex: 0.05 = 5%)
        take_profit: Percentual de take-profit (ex: 0.10 = 10%)
        max_position_size: Tamanho máximo da posição
        
    Returns:
        Tuple[pd.Series, pd.DataFrame]: Sinais ajustados e log de operações
    """
    adjusted_signals = pd.Series(0, index=prices.index)
    operations_log = []
    
    position = 0
    entry_price = 0
    entry_date = None
    
    for i in range(1, len(prices)):
        current_price = prices.iloc[i]
        current_date = prices.index[i]
        
        # Se não há posição, verificar sinais de entrada
        if position == 0:
            if signals.iloc[i] == 1:  # Sinal de compra
                position = max_position_size
                entry_price = current_price
                entry_date = current_date
                adjusted_signals.iloc[i] = 1
                
                operations_log.append({
                    'date': current_date,
                    'action': 'BUY',
                    'price': current_price,
                    'position': position,
                    'reason': 'SIGNAL'
                })
        
        # Se há posição, verificar condições de saída
        elif position > 0:
            return_pct = (current_price / entry_price) - 1
            
            # Verificar stop-loss
            if return_pct <= -stop_loss:
                position = 0
                adjusted_signals.iloc[i] = -1
                
                operations_log.append({
                    'date': current_date,
                    'action': 'SELL',
                    'price': current_price,
                    'position': 0,
                    'reason': 'STOP_LOSS',
                    'return': return_pct
                })
            
            # Verificar take-profit
            elif return_pct >= take_profit:
                position = 0
                adjusted_signals.iloc[i] = -1
                
                operations_log.append({
                    'date': current_date,
                    'action': 'SELL',
                    'price': current_price,
                    'position': 0,
                    'reason': 'TAKE_PROFIT',
                    'return': return_pct
                })
            
            # Verificar sinal de venda
            elif signals.iloc[i] == -1:
                position = 0
                adjusted_signals.iloc[i] = -1
                
                operations_log.append({
                    'date': current_date,
                    'action': 'SELL',
                    'price': current_price,
                    'position': 0,
                    'reason': 'SIGNAL',
                    'return': return_pct
                })
    
    operations_df = pd.DataFrame(operations_log)
    return adjusted_signals, operations_df

def advanced_strategy_simulation(prices: pd.Series, predictions: np.ndarray = None,
                               indicators: Dict[str, pd.Series] = None,
                               strategy_type: str = 'technical',
                               risk_management: bool = True,
                               **kwargs) -> Dict:
    """
    Simulação de estratégias avançadas com diferentes tipos e gestão de risco.
    
    Args:
        prices: Série de preços históricos
        predictions: Array de previsões (opcional)
        indicators: Dicionário com indicadores técnicos (opcional)
        strategy_type: Tipo de estratégia ('technical', 'momentum', 'mean_reversion')
        risk_management: Se deve aplicar gestão de risco
        **kwargs: Parâmetros adicionais para as estratégias
        
    Returns:
        Dict: Resultados da simulação
    """
    # Gerar sinais baseados no tipo de estratégia
    if strategy_type == 'technical' and indicators is not None:
        signals = technical_indicator_strategy(prices, indicators, predictions, **kwargs)
    elif strategy_type == 'momentum' and predictions is not None:
        signals = momentum_strategy(prices, predictions, **kwargs)
    elif strategy_type == 'mean_reversion':
        signals = mean_reversion_strategy(prices, **kwargs)
    else:
        # Fallback para estratégia simples baseada em previsões
        if predictions is not None:
            signals = generate_signals(prices, predictions)
        else:
            raise ValueError("Tipo de estratégia não suportado ou dados insuficientes")
    
    # Aplicar gestão de risco se solicitado
    operations_log = None
    if risk_management:
        signals, operations_log = risk_managed_strategy(prices, signals, **kwargs)
    
    # Simular retornos da estratégia
    returns = simulate_strategy_returns(prices, signals)
    
    # Calcular métricas de performance
    metrics = calculate_performance_metrics(signals, prices, kwargs.get('initial_capital', 10000), kwargs.get('transaction_cost', 0.001))
    
    # Preparar resultados
    results = {
        'signals': signals,
        'returns': returns,
        'metrics': metrics,
        'strategy_type': strategy_type,
        'risk_management': risk_management
    }
    
    if operations_log is not None:
        results['operations_log'] = operations_log
    
    return results

def simulate_strategy_returns(prices: pd.Series, signals: pd.Series) -> pd.Series:
    """
    Simula retornos baseados nos sinais de trading.
    
    Args:
        prices: Série de preços históricos
        signals: Sinais de trading
        
    Returns:
        pd.Series: Retornos da estratégia
    """
    returns = pd.Series(0.0, index=prices.index)
    position = 0
    
    for i in range(1, len(prices)):
        # Calcular retorno diário do ativo
        daily_return = prices.iloc[i] / prices.iloc[i-1] - 1
        
        # Atualizar posição baseada no sinal
        if signals.iloc[i-1] == 1:  # Sinal de compra
            position = 1
        elif signals.iloc[i-1] == -1:  # Sinal de venda
            position = 0
        
        # Calcular retorno da estratégia
        if position == 1:
            returns.iloc[i] = daily_return
        else:
            returns.iloc[i] = 0
    
    return returns

def compare_strategies(strategies, prices: pd.Series, predictions: np.ndarray = None,
                      indicators: Dict[str, pd.Series] = None, 
                      initial_capital: float = 10000,
                      transaction_cost: float = 0.001) -> Dict:
    """
    Compara diferentes estratégias de trading.
    
    Args:
        strategies: Lista de estratégias para comparar
        prices: Série de preços históricos
        predictions: Array de previsões (opcional)
        indicators: Dicionário com indicadores técnicos (opcional)
        initial_capital: Capital inicial para simulação
        transaction_cost: Custo de transação como percentual
        
    Returns:
        Dict: Comparação de métricas entre estratégias
    """
    strategies_results = {}
    
    # Buy and Hold
    bh_returns = simulate_buy_and_hold(prices)
    bh_signals = pd.Series(1, index=prices.index)  # Sempre comprado
    strategies_results['Buy & Hold'] = calculate_performance_metrics(bh_signals, prices, initial_capital, transaction_cost)
    
    # Estratégia técnica (se indicadores disponíveis)
    if indicators is not None:
        tech_results = advanced_strategy_simulation(
            prices, predictions, indicators, 'technical', risk_management=False
        )
        strategies_results['Technical'] = tech_results['metrics']
        
        tech_risk_results = advanced_strategy_simulation(
            prices, predictions, indicators, 'technical', risk_management=True
        )
        strategies_results['Technical + Risk Mgmt'] = tech_risk_results['metrics']
    
    # Estratégia de momentum (se previsões disponíveis)
    if predictions is not None:
        momentum_results = advanced_strategy_simulation(
            prices, predictions, indicators, 'momentum', risk_management=False
        )
        strategies_results['Momentum'] = momentum_results['metrics']
        
        momentum_risk_results = advanced_strategy_simulation(
            prices, predictions, indicators, 'momentum', risk_management=True
        )
        strategies_results['Momentum + Risk Mgmt'] = momentum_risk_results['metrics']
    
    # Estratégia de reversão à média
    mr_results = advanced_strategy_simulation(
        prices, predictions, indicators, 'mean_reversion', risk_management=False
    )
    strategies_results['Mean Reversion'] = mr_results['metrics']
    
    mr_risk_results = advanced_strategy_simulation(
        prices, predictions, indicators, 'mean_reversion', risk_management=True
    )
    strategies_results['Mean Reversion + Risk Mgmt'] = mr_risk_results['metrics']
    
    return strategies_results

# Exemplo de uso
if __name__ == "__main__":
    # Criar dados de exemplo
    dates = pd.date_range(start="2020-01-01", periods=100)
    prices = pd.Series(100 + np.random.randn(100).cumsum(), index=dates)
    
    # Criar previsões simuladas
    forecast_horizon = 14
    n_samples = len(prices)
    predictions = np.zeros((n_samples, forecast_horizon))
    
    for i in range(n_samples):
        current_price = prices.iloc[i]
        # Simular previsões com tendência de alta
        predictions[i] = current_price * (1 + np.random.randn(forecast_horizon) * 0.01 + 0.005)
    
    # Simular estratégias
    long_only_returns = simulate_long_only_strategy(prices, predictions, forecast_horizon)
    buy_hold_returns = simulate_buy_and_hold(prices)
    
    # Calcular métricas
    long_only_metrics = calculate_performance_metrics(long_only_returns)
    buy_hold_metrics = calculate_performance_metrics(buy_hold_returns)
    
    # Imprimir resultados
    print("Métricas da Estratégia Long-Only:")
    for metric, value in long_only_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nMétricas da Estratégia Buy-and-Hold:")
    for metric, value in buy_hold_metrics.items():
        print(f"  {metric}: {value:.4f}")

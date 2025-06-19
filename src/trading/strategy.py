"""Módulo de estratégias de trading baseadas em indicadores técnicos.

Este módulo implementa estratégias de trading conforme o documento de qualidade,
utilizando os indicadores técnicos corretos e variáveis exógenas.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from enum import Enum


class SignalType(Enum):
    """Tipos de sinais de trading."""
    BUY = 1
    SELL = -1
    HOLD = 0


class TradingStrategy:
    """Classe base para estratégias de trading."""
    
    def __init__(self, name: str):
        self.name = name
        self.signals = []
        self.positions = []
        self.returns = []
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Gera sinais de trading baseados no DataFrame.
        
        Args:
            df (pd.DataFrame): DataFrame com dados e indicadores.
            
        Returns:
            pd.Series: Série com sinais de trading.
        """
        raise NotImplementedError("Subclasses devem implementar este método")
    
    def calculate_returns(self, df: pd.DataFrame, signals: pd.Series) -> pd.Series:
        """Calcula retornos baseados nos sinais.
        
        Args:
            df (pd.DataFrame): DataFrame com dados de preços.
            signals (pd.Series): Sinais de trading.
            
        Returns:
            pd.Series: Retornos da estratégia.
        """
        # Calcular retornos do ativo
        asset_returns = df['Adj Close'].pct_change()
        
        # Aplicar sinais (shift para evitar look-ahead bias)
        strategy_returns = signals.shift(1) * asset_returns
        
        return strategy_returns.fillna(0)


class BollingerBandsStrategy(TradingStrategy):
    """Estratégia baseada em Bollinger Bands."""
    
    def __init__(self, oversold_threshold: float = 0.1, overbought_threshold: float = 0.9):
        super().__init__("Bollinger Bands Strategy")
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Gera sinais baseados na posição do preço nas Bollinger Bands."""
        signals = pd.Series(0, index=df.index)
        
        if 'BB_Position' not in df.columns:
            print("Aviso: BB_Position não encontrado no DataFrame")
            return signals
        
        # Sinal de compra quando preço está próximo da banda inferior
        buy_condition = df['BB_Position'] <= self.oversold_threshold
        
        # Sinal de venda quando preço está próximo da banda superior
        sell_condition = df['BB_Position'] >= self.overbought_threshold
        
        signals[buy_condition] = SignalType.BUY.value
        signals[sell_condition] = SignalType.SELL.value
        
        return signals


class StochasticStrategy(TradingStrategy):
    """Estratégia baseada no Oscilador Estocástico."""
    
    def __init__(self, oversold_level: float = 20, overbought_level: float = 80):
        super().__init__("Stochastic Strategy")
        self.oversold_level = oversold_level
        self.overbought_level = overbought_level
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Gera sinais baseados no Oscilador Estocástico."""
        signals = pd.Series(0, index=df.index)
        
        if 'Stoch_K' not in df.columns or 'Stoch_D' not in df.columns:
            print("Aviso: Indicadores Stochastic não encontrados no DataFrame")
            return signals
        
        # Sinal de compra: %K cruza %D para cima em região de sobrevenda
        buy_condition = (
            (df['Stoch_K'] > df['Stoch_D']) & 
            (df['Stoch_K'].shift(1) <= df['Stoch_D'].shift(1)) &
            (df['Stoch_K'] < self.oversold_level)
        )
        
        # Sinal de venda: %K cruza %D para baixo em região de sobrecompra
        sell_condition = (
            (df['Stoch_K'] < df['Stoch_D']) & 
            (df['Stoch_K'].shift(1) >= df['Stoch_D'].shift(1)) &
            (df['Stoch_K'] > self.overbought_level)
        )
        
        signals[buy_condition] = SignalType.BUY.value
        signals[sell_condition] = SignalType.SELL.value
        
        return signals


class ADXTrendStrategy(TradingStrategy):
    """Estratégia baseada no ADX para identificação de tendências."""
    
    def __init__(self, adx_threshold: float = 25):
        super().__init__("ADX Trend Strategy")
        self.adx_threshold = adx_threshold
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Gera sinais baseados na força da tendência (ADX)."""
        signals = pd.Series(0, index=df.index)
        
        required_cols = ['ADX', 'ADX_Pos', 'ADX_Neg']
        if not all(col in df.columns for col in required_cols):
            print(f"Aviso: Indicadores ADX não encontrados no DataFrame")
            return signals
        
        # Tendência forte quando ADX > threshold
        strong_trend = df['ADX'] > self.adx_threshold
        
        # Sinal de compra: tendência forte e +DI > -DI
        buy_condition = strong_trend & (df['ADX_Pos'] > df['ADX_Neg'])
        
        # Sinal de venda: tendência forte e -DI > +DI
        sell_condition = strong_trend & (df['ADX_Neg'] > df['ADX_Pos'])
        
        signals[buy_condition] = SignalType.BUY.value
        signals[sell_condition] = SignalType.SELL.value
        
        return signals


class WilliamsRStrategy(TradingStrategy):
    """Estratégia baseada no Williams %R."""
    
    def __init__(self, oversold_level: float = -80, overbought_level: float = -20):
        super().__init__("Williams %R Strategy")
        self.oversold_level = oversold_level
        self.overbought_level = overbought_level
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Gera sinais baseados no Williams %R."""
        signals = pd.Series(0, index=df.index)
        
        if 'Williams_R' not in df.columns:
            print("Aviso: Williams_R não encontrado no DataFrame")
            return signals
        
        # Sinal de compra: saída da região de sobrevenda
        buy_condition = (
            (df['Williams_R'] > self.oversold_level) &
            (df['Williams_R'].shift(1) <= self.oversold_level)
        )
        
        # Sinal de venda: entrada na região de sobrecompra
        sell_condition = (
            (df['Williams_R'] < self.overbought_level) &
            (df['Williams_R'].shift(1) >= self.overbought_level)
        )
        
        signals[buy_condition] = SignalType.BUY.value
        signals[sell_condition] = SignalType.SELL.value
        
        return signals


class MultiIndicatorStrategy(TradingStrategy):
    """Estratégia que combina múltiplos indicadores técnicos."""
    
    def __init__(self, weights: Dict[str, float] = None):
        super().__init__("Multi-Indicator Strategy")
        self.weights = weights or {
            'bollinger': 0.25,
            'stochastic': 0.25,
            'williams': 0.20,
            'adx': 0.15,
            'cci': 0.15
        }
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Gera sinais combinando múltiplos indicadores."""
        signals = pd.Series(0.0, index=df.index)
        
        # Bollinger Bands
        if 'BB_Position' in df.columns:
            bb_signals = pd.Series(0.0, index=df.index)
            bb_signals[df['BB_Position'] <= 0.1] = 1.0  # Compra
            bb_signals[df['BB_Position'] >= 0.9] = -1.0  # Venda
            signals += bb_signals * self.weights['bollinger']
        
        # Stochastic
        if all(col in df.columns for col in ['Stoch_K', 'Stoch_D']):
            stoch_signals = pd.Series(0.0, index=df.index)
            # Compra quando %K cruza %D para cima em sobrevenda
            buy_stoch = (
                (df['Stoch_K'] > df['Stoch_D']) & 
                (df['Stoch_K'].shift(1) <= df['Stoch_D'].shift(1)) &
                (df['Stoch_K'] < 20)
            )
            # Venda quando %K cruza %D para baixo em sobrecompra
            sell_stoch = (
                (df['Stoch_K'] < df['Stoch_D']) & 
                (df['Stoch_K'].shift(1) >= df['Stoch_D'].shift(1)) &
                (df['Stoch_K'] > 80)
            )
            stoch_signals[buy_stoch] = 1.0
            stoch_signals[sell_stoch] = -1.0
            signals += stoch_signals * self.weights['stochastic']
        
        # Williams %R
        if 'Williams_R' in df.columns:
            williams_signals = pd.Series(0.0, index=df.index)
            williams_signals[df['Williams_R'] > -20] = -1.0  # Sobrecompra
            williams_signals[df['Williams_R'] < -80] = 1.0   # Sobrevenda
            signals += williams_signals * self.weights['williams']
        
        # ADX Trend
        if all(col in df.columns for col in ['ADX', 'ADX_Pos', 'ADX_Neg']):
            adx_signals = pd.Series(0.0, index=df.index)
            strong_trend = df['ADX'] > 25
            adx_signals[strong_trend & (df['ADX_Pos'] > df['ADX_Neg'])] = 1.0
            adx_signals[strong_trend & (df['ADX_Neg'] > df['ADX_Pos'])] = -1.0
            signals += adx_signals * self.weights['adx']
        
        # CCI
        if 'CCI' in df.columns:
            cci_signals = pd.Series(0.0, index=df.index)
            cci_signals[df['CCI'] < -100] = 1.0   # Sobrevenda
            cci_signals[df['CCI'] > 100] = -1.0   # Sobrecompra
            signals += cci_signals * self.weights['cci']
        
        # Converter para sinais discretos
        final_signals = pd.Series(0, index=df.index)
        final_signals[signals > 0.3] = SignalType.BUY.value
        final_signals[signals < -0.3] = SignalType.SELL.value
        
        return final_signals


class ExogenousEnhancedStrategy(TradingStrategy):
    """Estratégia que incorpora variáveis exógenas."""
    
    def __init__(self):
        super().__init__("Exogenous Enhanced Strategy")
    
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Gera sinais considerando variáveis exógenas."""
        signals = pd.Series(0, index=df.index)
        
        # Começar com sinais dos indicadores técnicos
        multi_strategy = MultiIndicatorStrategy()
        base_signals = multi_strategy.generate_signals(df)
        
        # Modificar sinais baseado em variáveis exógenas
        
        # 1. Filtro de volatilidade (VIX)
        if 'VIX_Close' in df.columns:
            # Reduzir exposição quando VIX muito alto (volatilidade extrema)
            high_vix = df['VIX_Close'] > 30
            base_signals[high_vix] = base_signals[high_vix] * 0.5
        
        # 2. Filtro de mercado (SPY)
        if 'SPY_Return' in df.columns:
            # Evitar posições longas quando mercado em queda
            market_down = df['SPY_Return'] < -0.02  # Queda > 2%
            base_signals[(base_signals > 0) & market_down] = 0
        
        # 3. Filtro de força do dólar (DXY)
        if 'DXY_Strength' in df.columns:
            # Para ativos que são negativamente correlacionados com DXY
            strong_dollar = df['DXY_Strength'] == 1
            # Reduzir sinais de compra quando dólar forte
            base_signals[(base_signals > 0) & strong_dollar] = base_signals[(base_signals > 0) & strong_dollar] * 0.7
        
        # 4. Filtro de risco de mercado
        if 'Market_Risk_Level' in df.columns:
            # Reduzir exposição em níveis altos de risco
            high_risk = df['Market_Risk_Level'] >= 2
            base_signals[high_risk] = base_signals[high_risk] * 0.6
        
        return base_signals.astype(int)


class StrategyBacktester:
    """Classe para backtesting de estratégias."""
    
    def __init__(self, initial_capital: float = 100000, 
                 transaction_cost: float = 0.001):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
    
    def backtest_strategy(self, df: pd.DataFrame, strategy: TradingStrategy) -> Dict:
        """Executa backtest de uma estratégia.
        
        Args:
            df (pd.DataFrame): DataFrame com dados e indicadores.
            strategy (TradingStrategy): Estratégia a ser testada.
            
        Returns:
            Dict: Resultados do backtest.
        """
        # Gerar sinais
        signals = strategy.generate_signals(df)
        
        # Calcular retornos
        strategy_returns = strategy.calculate_returns(df, signals)
        
        # Aplicar custos de transação
        position_changes = signals.diff().abs()
        transaction_costs = position_changes * self.transaction_cost
        net_returns = strategy_returns - transaction_costs
        
        # Calcular métricas
        cumulative_returns = (1 + net_returns).cumprod()
        total_return = cumulative_returns.iloc[-1] - 1
        
        # Sharpe Ratio
        sharpe_ratio = np.sqrt(252) * net_returns.mean() / net_returns.std() if net_returns.std() > 0 else 0
        
        # Maximum Drawdown
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Número de trades
        num_trades = position_changes.sum() / 2  # Dividir por 2 pois cada trade tem entrada e saída
        
        # Win rate
        winning_trades = net_returns[net_returns > 0]
        win_rate = len(winning_trades) / len(net_returns[net_returns != 0]) if len(net_returns[net_returns != 0]) > 0 else 0
        
        return {
            'strategy_name': strategy.name,
            'total_return': total_return,
            'annualized_return': (1 + total_return) ** (252 / len(df)) - 1,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': abs(max_drawdown),
            'num_trades': num_trades,
            'win_rate': win_rate,
            'volatility': net_returns.std() * np.sqrt(252),
            'signals': signals,
            'returns': net_returns,
            'cumulative_returns': cumulative_returns
        }
    
    def compare_strategies(self, df: pd.DataFrame, 
                          strategies: List[TradingStrategy]) -> pd.DataFrame:
        """Compara múltiplas estratégias.
        
        Args:
            df (pd.DataFrame): DataFrame com dados e indicadores.
            strategies (List[TradingStrategy]): Lista de estratégias.
            
        Returns:
            pd.DataFrame: Comparação das estratégias.
        """
        results = []
        
        for strategy in strategies:
            result = self.backtest_strategy(df, strategy)
            results.append({
                'Strategy': result['strategy_name'],
                'Total Return': f"{result['total_return']:.2%}",
                'Annualized Return': f"{result['annualized_return']:.2%}",
                'Sharpe Ratio': f"{result['sharpe_ratio']:.3f}",
                'Max Drawdown': f"{result['max_drawdown']:.2%}",
                'Volatility': f"{result['volatility']:.2%}",
                'Num Trades': int(result['num_trades']),
                'Win Rate': f"{result['win_rate']:.2%}"
            })
        
        return pd.DataFrame(results)


if __name__ == "__main__":
    # Exemplo de uso
    print("Módulo de Estratégias de Trading carregado com sucesso!")
    print("Estratégias disponíveis:")
    print("- BollingerBandsStrategy")
    print("- StochasticStrategy")
    print("- ADXTrendStrategy")
    print("- WilliamsRStrategy")
    print("- MultiIndicatorStrategy")
    print("- ExogenousEnhancedStrategy")
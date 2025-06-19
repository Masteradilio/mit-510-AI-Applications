"""Módulo para coleta e processamento de variáveis exógenas.

Este módulo implementa a coleta de dados macroeconômicos, índices de mercado,
commodities e indicadores de sentimento que podem influenciar os preços
dos ativos principais (BTC e AAPL).
"""

import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
import numpy as np
from typing import Optional, Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class ExogenousDataCollector:
    """Coletor de dados exógenos para análise financeira.
    
    Esta classe centraliza a coleta de diferentes tipos de variáveis exógenas
    que podem impactar os preços dos ativos financeiros.
    """
    
    def __init__(self):
        """Inicializa o coletor de dados exógenos."""
        self.data_cache = {}
        
    def get_market_indices(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Coleta dados de índices de mercado.
        
        Args:
            start_date (str, optional): Data de início (YYYY-MM-DD)
            end_date (str, optional): Data de fim (YYYY-MM-DD)
            
        Returns:
            pd.DataFrame: DataFrame com dados dos índices de mercado
        """
        indices = {
            'SPY': '^GSPC',  # S&P 500
            'NASDAQ': '^IXIC',  # NASDAQ Composite
            'VIX': '^VIX',  # Volatility Index
            'DJI': '^DJI',  # Dow Jones Industrial Average
            'RUSSELL2000': '^RUT'  # Russell 2000
        }
        
        market_data = pd.DataFrame()
        
        for name, ticker in indices.items():
            try:
                print(f"Coletando dados do índice {name} ({ticker})...")
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if not data.empty:
                    # Usar apenas o preço de fechamento ajustado
                    if 'Adj Close' in data.columns:
                        market_data[f'{name}_Close'] = data['Adj Close']
                    else:
                        market_data[f'{name}_Close'] = data['Close']
                    
                    # Adicionar volume se disponível
                    if 'Volume' in data.columns:
                        market_data[f'{name}_Volume'] = data['Volume']
                        
                    # Calcular retornos diários
                    market_data[f'{name}_Returns'] = market_data[f'{name}_Close'].pct_change()
                    
                    # Calcular volatilidade móvel (20 dias)
                    market_data[f'{name}_Volatility'] = market_data[f'{name}_Returns'].rolling(20).std()
                    
            except Exception as e:
                print(f"Erro ao coletar dados do índice {name}: {e}")
                continue
                
        # Garantir que o índice seja tz-naive
        if market_data.index.tz is not None:
            market_data.index = market_data.index.tz_localize(None)
            
        return market_data
    
    def get_commodities_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Coleta dados de commodities.
        
        Args:
            start_date (str, optional): Data de início (YYYY-MM-DD)
            end_date (str, optional): Data de fim (YYYY-MM-DD)
            
        Returns:
            pd.DataFrame: DataFrame com dados de commodities
        """
        commodities = {
            'GOLD': 'GC=F',  # Gold Futures
            'SILVER': 'SI=F',  # Silver Futures
            'OIL_WTI': 'CL=F',  # Crude Oil WTI Futures
            'OIL_BRENT': 'BZ=F',  # Brent Crude Oil Futures
            'COPPER': 'HG=F',  # Copper Futures
            'NATURAL_GAS': 'NG=F'  # Natural Gas Futures
        }
        
        commodities_data = pd.DataFrame()
        
        for name, ticker in commodities.items():
            try:
                print(f"Coletando dados da commodity {name} ({ticker})...")
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if not data.empty:
                    # Usar preço de fechamento ajustado
                    if 'Adj Close' in data.columns:
                        commodities_data[f'{name}_Close'] = data['Adj Close']
                    else:
                        commodities_data[f'{name}_Close'] = data['Close']
                    
                    # Adicionar volume se disponível
                    if 'Volume' in data.columns:
                        commodities_data[f'{name}_Volume'] = data['Volume']
                        
                    # Calcular retornos diários
                    commodities_data[f'{name}_Returns'] = commodities_data[f'{name}_Close'].pct_change()
                    
                    # Calcular volatilidade móvel (20 dias)
                    commodities_data[f'{name}_Volatility'] = commodities_data[f'{name}_Returns'].rolling(20).std()
                    
            except Exception as e:
                print(f"Erro ao coletar dados da commodity {name}: {e}")
                continue
                
        # Garantir que o índice seja tz-naive
        if commodities_data.index.tz is not None:
            commodities_data.index = commodities_data.index.tz_localize(None)
            
        return commodities_data
    
    def get_currency_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Coleta dados de moedas e criptomoedas.
        
        Args:
            start_date (str, optional): Data de início (YYYY-MM-DD)
            end_date (str, optional): Data de fim (YYYY-MM-DD)
            
        Returns:
            pd.DataFrame: DataFrame com dados de moedas
        """
        currencies = {
            'USD_EUR': 'EURUSD=X',  # Euro/USD
            'USD_JPY': 'JPY=X',     # USD/Japanese Yen
            'USD_GBP': 'GBPUSD=X',  # British Pound/USD
            'USD_CHF': 'CHF=X',     # USD/Swiss Franc
            'DXY': 'DX-Y.NYB',      # US Dollar Index
            'ETH': 'ETH-USD',       # Ethereum (como proxy crypto)
        }
        
        currency_data = pd.DataFrame()
        
        for name, ticker in currencies.items():
            try:
                print(f"Coletando dados da moeda {name} ({ticker})...")
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if not data.empty:
                    # Usar preço de fechamento ajustado
                    if 'Adj Close' in data.columns:
                        currency_data[f'{name}_Close'] = data['Adj Close']
                    else:
                        currency_data[f'{name}_Close'] = data['Close']
                    
                    # Calcular retornos diários
                    currency_data[f'{name}_Returns'] = currency_data[f'{name}_Close'].pct_change()
                    
                    # Calcular volatilidade móvel (20 dias)
                    currency_data[f'{name}_Volatility'] = currency_data[f'{name}_Returns'].rolling(20).std()
                    
            except Exception as e:
                print(f"Erro ao coletar dados da moeda {name}: {e}")
                continue
                
        # Garantir que o índice seja tz-naive
        if currency_data.index.tz is not None:
            currency_data.index = currency_data.index.tz_localize(None)
            
        return currency_data
    
    def get_treasury_rates(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Coleta dados de taxas de juros do Tesouro Americano.
        
        Args:
            start_date (str, optional): Data de início (YYYY-MM-DD)
            end_date (str, optional): Data de fim (YYYY-MM-DD)
            
        Returns:
            pd.DataFrame: DataFrame com taxas de juros
        """
        treasury_rates = {
            'TREASURY_10Y': '^TNX',    # 10-Year Treasury
            'TREASURY_2Y': '^IRX',     # 13-Week Treasury Bill
            'TREASURY_30Y': '^TYX',    # 30-Year Treasury
        }
        
        rates_data = pd.DataFrame()
        
        for name, ticker in treasury_rates.items():
            try:
                print(f"Coletando dados da taxa {name} ({ticker})...")
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if not data.empty:
                    # Usar preço de fechamento (que representa a taxa)
                    rates_data[f'{name}_Rate'] = data['Close']
                    
                    # Calcular mudanças diárias na taxa
                    rates_data[f'{name}_Change'] = rates_data[f'{name}_Rate'].diff()
                    
            except Exception as e:
                print(f"Erro ao coletar dados da taxa {name}: {e}")
                continue
                
        # Calcular spread entre taxas
        if 'TREASURY_10Y_Rate' in rates_data.columns and 'TREASURY_2Y_Rate' in rates_data.columns:
            rates_data['YIELD_CURVE_SPREAD'] = rates_data['TREASURY_10Y_Rate'] - rates_data['TREASURY_2Y_Rate']
            
        # Garantir que o índice seja tz-naive
        if rates_data.index.tz is not None:
            rates_data.index = rates_data.index.tz_localize(None)
            
        return rates_data
    
    def get_sentiment_indicators(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Coleta indicadores de sentimento de mercado.
        
        Args:
            start_date (str, optional): Data de início (YYYY-MM-DD)
            end_date (str, optional): Data de fim (YYYY-MM-DD)
            
        Returns:
            pd.DataFrame: DataFrame com indicadores de sentimento
        """
        sentiment_data = pd.DataFrame()
        
        # Usar VIX como proxy para Fear & Greed (já coletado em market_indices)
        # Aqui vamos criar indicadores derivados
        try:
            print("Coletando dados de sentimento (VIX)...")
            vix_data = yf.download('^VIX', start=start_date, end=end_date, progress=False)
            
            if not vix_data.empty:
                sentiment_data['VIX_Level'] = vix_data['Close']
                
                # Criar categorias de sentimento baseadas no VIX
                # VIX < 20: Low Fear (Greed)
                # VIX 20-30: Moderate Fear
                # VIX > 30: High Fear
                sentiment_data['Fear_Level'] = pd.cut(
                    sentiment_data['VIX_Level'],
                    bins=[0, 20, 30, 100],
                    labels=['Low_Fear', 'Moderate_Fear', 'High_Fear']
                )
                
                # Converter para variáveis dummy
                fear_dummies = pd.get_dummies(sentiment_data['Fear_Level'], prefix='Fear')
                sentiment_data = pd.concat([sentiment_data, fear_dummies], axis=1)
                
                # Calcular média móvel do VIX (indicador de tendência do medo)
                sentiment_data['VIX_MA_20'] = sentiment_data['VIX_Level'].rolling(20).mean()
                sentiment_data['VIX_Trend'] = sentiment_data['VIX_Level'] - sentiment_data['VIX_MA_20']
                
        except Exception as e:
            print(f"Erro ao coletar dados de sentimento: {e}")
            
        # Adicionar indicadores baseados em outros ativos
        try:
            # Put/Call Ratio aproximado usando ETFs
            print("Coletando dados de Put/Call ratio...")
            spy_data = yf.download('SPY', start=start_date, end=end_date, progress=False)
            
            if not spy_data.empty:
                # Calcular volatilidade realizada como proxy para nervosismo
                spy_returns = spy_data['Close'].pct_change()
                sentiment_data['SPY_Realized_Vol'] = spy_returns.rolling(20).std() * np.sqrt(252)
                
                # Calcular RSI do SPY como indicador de sentimento
                delta = spy_data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                sentiment_data['SPY_RSI'] = 100 - (100 / (1 + rs))
                
        except Exception as e:
            print(f"Erro ao calcular indicadores de sentimento derivados: {e}")
            
        # Garantir que o índice seja tz-naive
        if sentiment_data.index.tz is not None:
            sentiment_data.index = sentiment_data.index.tz_localize(None)
            
        return sentiment_data
    
    def get_economic_indicators(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Coleta indicadores econômicos usando proxies de mercado.
        
        Args:
            start_date (str, optional): Data de início (YYYY-MM-DD)
            end_date (str, optional): Data de fim (YYYY-MM-DD)
            
        Returns:
            pd.DataFrame: DataFrame com indicadores econômicos
        """
        economic_data = pd.DataFrame()
        
        # Usar ETFs como proxies para setores econômicos
        economic_proxies = {
            'FINANCIALS': 'XLF',    # Financial Select Sector SPDR Fund
            'TECHNOLOGY': 'XLK',    # Technology Select Sector SPDR Fund
            'ENERGY': 'XLE',        # Energy Select Sector SPDR Fund
            'HEALTHCARE': 'XLV',    # Health Care Select Sector SPDR Fund
            'CONSUMER_DISC': 'XLY', # Consumer Discretionary Select Sector SPDR Fund
            'UTILITIES': 'XLU',     # Utilities Select Sector SPDR Fund
        }
        
        for name, ticker in economic_proxies.items():
            try:
                print(f"Coletando dados do setor {name} ({ticker})...")
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if not data.empty:
                    # Usar preço de fechamento ajustado
                    if 'Adj Close' in data.columns:
                        economic_data[f'{name}_Close'] = data['Adj Close']
                    else:
                        economic_data[f'{name}_Close'] = data['Close']
                    
                    # Calcular retornos diários
                    economic_data[f'{name}_Returns'] = economic_data[f'{name}_Close'].pct_change()
                    
                    # Calcular performance relativa ao S&P 500
                    spy_data = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
                    if not spy_data.empty:
                        spy_returns = spy_data['Close'].pct_change()
                        economic_data[f'{name}_Relative_Performance'] = (
                            economic_data[f'{name}_Returns'] - spy_returns
                        ).rolling(20).mean()
                    
            except Exception as e:
                print(f"Erro ao coletar dados do setor {name}: {e}")
                continue
                
        # Garantir que o índice seja tz-naive
        if economic_data.index.tz is not None:
            economic_data.index = economic_data.index.tz_localize(None)
            
        return economic_data
    
    def consolidate_exogenous_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Consolida todos os dados exógenos em um único DataFrame.
        
        Args:
            start_date (str, optional): Data de início (YYYY-MM-DD)
            end_date (str, optional): Data de fim (YYYY-MM-DD)
            
        Returns:
            pd.DataFrame: DataFrame consolidado com todas as variáveis exógenas
        """
        print("=== Iniciando coleta de dados exógenos ===")
        
        # Coletar todos os tipos de dados
        market_data = self.get_market_indices(start_date, end_date)
        commodities_data = self.get_commodities_data(start_date, end_date)
        currency_data = self.get_currency_data(start_date, end_date)
        rates_data = self.get_treasury_rates(start_date, end_date)
        sentiment_data = self.get_sentiment_indicators(start_date, end_date)
        economic_data = self.get_economic_indicators(start_date, end_date)
        
        # Lista de DataFrames para consolidar
        dataframes = [
            market_data, commodities_data, currency_data, 
            rates_data, sentiment_data, economic_data
        ]
        
        # Filtrar DataFrames não vazios
        non_empty_dfs = [df for df in dataframes if not df.empty]
        
        if not non_empty_dfs:
            print("Aviso: Nenhum dado exógeno foi coletado com sucesso.")
            return pd.DataFrame()
        
        # Consolidar todos os dados
        print("Consolidando dados exógenos...")
        consolidated_data = pd.concat(non_empty_dfs, axis=1, sort=True)
        
        # Remover colunas duplicadas (se houver)
        consolidated_data = consolidated_data.loc[:, ~consolidated_data.columns.duplicated()]
        
        # Ordenar por data
        consolidated_data.sort_index(inplace=True)
        
        print(f"Dados exógenos consolidados: {consolidated_data.shape[0]} observações, {consolidated_data.shape[1]} variáveis")
        print(f"Período: {consolidated_data.index.min()} a {consolidated_data.index.max()}")
        
        return consolidated_data
    
    def synchronize_with_asset_data(self, asset_data: pd.DataFrame, exogenous_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Sincroniza dados exógenos com dados do ativo principal.
        
        Args:
            asset_data (pd.DataFrame): DataFrame com dados do ativo principal
            exogenous_data (pd.DataFrame): DataFrame com dados exógenos
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Tupla com (asset_data_sync, exogenous_data_sync)
        """
        if asset_data.empty or exogenous_data.empty:
            print("Aviso: Um dos DataFrames está vazio. Retornando dados originais.")
            return asset_data, exogenous_data
        
        # Garantir que ambos os índices sejam tz-naive
        if asset_data.index.tz is not None:
            asset_data.index = asset_data.index.tz_localize(None)
        if exogenous_data.index.tz is not None:
            exogenous_data.index = exogenous_data.index.tz_localize(None)
        
        # Encontrar datas comuns
        common_dates = asset_data.index.intersection(exogenous_data.index)
        
        if len(common_dates) == 0:
            print("Aviso: Nenhuma data comum encontrada entre os datasets.")
            return asset_data, exogenous_data
        
        # Filtrar para datas comuns
        asset_sync = asset_data.loc[common_dates].copy()
        exogenous_sync = exogenous_data.loc[common_dates].copy()
        
        # Forward fill para preencher valores ausentes (dados de fim de semana, feriados)
        exogenous_sync = exogenous_sync.fillna(method='ffill')
        
        # Backward fill para valores ainda ausentes no início
        exogenous_sync = exogenous_sync.fillna(method='bfill')
        
        print(f"Sincronização concluída: {len(common_dates)} datas comuns")
        print(f"Período sincronizado: {common_dates.min()} a {common_dates.max()}")
        
        return asset_sync, exogenous_sync


# Função de conveniência para uso direto
def collect_exogenous_data(start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Função de conveniência para coletar todos os dados exógenos.
    
    Args:
        start_date (str, optional): Data de início (YYYY-MM-DD)
        end_date (str, optional): Data de fim (YYYY-MM-DD)
        
    Returns:
        pd.DataFrame: DataFrame com todos os dados exógenos consolidados
    """
    collector = ExogenousDataCollector()
    return collector.consolidate_exogenous_data(start_date, end_date)


# Exemplo de uso
if __name__ == '__main__':
    # Testar a coleta de dados exógenos
    print("Testando coleta de dados exógenos...")
    
    # Definir período de teste (últimos 2 anos)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    
    # Coletar dados
    collector = ExogenousDataCollector()
    exogenous_data = collector.consolidate_exogenous_data(start_date, end_date)
    
    if not exogenous_data.empty:
        print(f"\nDados coletados com sucesso!")
        print(f"Shape: {exogenous_data.shape}")
        print(f"\nPrimeiras 5 linhas:")
        print(exogenous_data.head())
        print(f"\nÚltimas 5 linhas:")
        print(exogenous_data.tail())
        print(f"\nColunas disponíveis:")
        print(exogenous_data.columns.tolist())
        print(f"\nEstatísticas descritivas:")
        print(exogenous_data.describe())
    else:
        print("Nenhum dado foi coletado.")
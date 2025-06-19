# -*- coding: utf-8 -*-
"""Teste de integração para o fluxo completo do aplicativo Streamlit."""

import os
import sys
import unittest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

# Adicionar diretório raiz ao path
test_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(test_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Importar módulos necessários
from streamlit.testing.v1 import AppTest
# unittest.mock.patch já está importado globalmente, MagicMock também.

# Os módulos do projeto e streamlit são importados após a configuração do sys.path
# e mocks globais, se necessário, mas AppTest geralmente lida bem com o st real.
from src.data_ingestion import loader
from src.preprocessing import feature_engineering, scalers_transformers
from src.modeling import prediction, strategy_simulation
from src.app.components import plotting, ui_elements
import streamlit as st # Confirmando que st é importado para uso com AppTest

# Caminho para os scripts do aplicativo Streamlit
APP_DIR = os.path.join(project_root, "src", "app")
HOME_PAGE_SCRIPT = os.path.join(APP_DIR, "Home.py")
PREDICTION_PAGE_SCRIPT = os.path.join(APP_DIR, "pages", "01_Previsão_de_Preços.py")
EXPLORATORY_PAGE_SCRIPT = os.path.join(APP_DIR, "pages", "02_Análise_Exploratória.py")

class TestAppIntegration(unittest.TestCase):
    """Testes de integração para o fluxo completo do aplicativo."""
    
    @classmethod
    def setUpClass(cls):
        """Configura o ambiente de teste uma vez para toda a classe."""
        cls.test_data_dir_class = os.path.join(project_root, "tests", "temp_data_class")
        cls.test_processed_dir_class = os.path.join(cls.test_data_dir_class, "processed")
        cls.test_models_dir_class = os.path.join(project_root, "tests", "temp_models_class")
        
        os.makedirs(cls.test_data_dir_class, exist_ok=True)
        os.makedirs(cls.test_processed_dir_class, exist_ok=True)
        os.makedirs(cls.test_models_dir_class, exist_ok=True)
        
        # Criar dados de teste que serão usados por múltiplos testes
        cls.btc_df_class, cls.aapl_df_class = cls._create_test_data_frames_class()
        cls.btc_df_class.to_csv(os.path.join(cls.test_processed_dir_class, "btc_processed_class.csv"))
        cls.aapl_df_class.to_csv(os.path.join(cls.test_processed_dir_class, "aapl_processed_class.csv"))
        cls._create_mock_models_scalers_class()

        # Patch global para run_ingestion_pipeline para testes que não são AppTest ou que precisam de dados mockados consistentes
        cls.global_load_data_patcher = patch('src.data_ingestion.loader.run_ingestion_pipeline', side_effect=cls._mocked_load_data_success_class)
        cls.mock_global_load_data = cls.global_load_data_patcher.start()

    @classmethod
    def tearDownClass(cls):
        """Limpa o ambiente de teste após todos os testes da classe."""
        cls.global_load_data_patcher.stop()
        # Remover arquivos e diretórios temporários da classe
        for dir_path in [cls.test_processed_dir_class, cls.test_models_dir_class]:
            for file_name in os.listdir(dir_path):
                try:
                    os.remove(os.path.join(dir_path, file_name))
                except OSError as e:
                    print(f"Erro ao remover arquivo {file_name} de {dir_path}: {e}")
            try:
                os.rmdir(dir_path)
            except OSError as e:
                 print(f"Erro ao remover diretório {dir_path}: {e}")
        try:
            os.rmdir(cls.test_data_dir_class)
        except OSError as e:
            print(f"Erro ao remover diretório {cls.test_data_dir_class}: {e}")

    @staticmethod
    def _create_test_data_frames_class():
        dates = pd.date_range(start="2023-01-01", periods=200)
        btc_df = pd.DataFrame({
            "Open": 20000 + np.random.randn(200).cumsum(), "High": 21000 + np.random.randn(200).cumsum(),
            "Low": 19000 + np.random.randn(200).cumsum(), "Close": 20500 + np.random.randn(200).cumsum(),
            "Adj Close": 20500 + np.random.randn(200).cumsum(), "Volume": np.random.randint(1000000, 5000000, 200)
        }, index=dates)
        aapl_df = pd.DataFrame({
            "Open": 150 + np.random.randn(200).cumsum(), "High": 155 + np.random.randn(200).cumsum(),
            "Low": 145 + np.random.randn(200).cumsum(), "Close": 152 + np.random.randn(200).cumsum(),
            "Adj Close": 152 + np.random.randn(200).cumsum(), "Volume": np.random.randint(10000000, 50000000, 200)
        }, index=dates)
        return btc_df, aapl_df

    @classmethod
    def _create_mock_models_scalers_class(cls):
        import tensorflow as tf
        import joblib
        from sklearn.preprocessing import MinMaxScaler
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(60, 6)), # Ajustar input_shape se necessário
            tf.keras.layers.Dense(14) # Ajustar output_shape se necessário
        ])
        model.compile(optimizer='adam', loss='mse')
        scaler = MinMaxScaler()
        # Usar dados de exemplo para o scaler, pode ser do btc_df_class
        # Certifique-se que as colunas usadas para fit são as mesmas que no código real
        # Exemplo: scaler.fit(cls.btc_df_class[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].values)
        # Para simplificar, vamos usar um fit genérico se os dados exatos não importarem tanto para o mock
        dummy_data_for_scaler = np.random.rand(100, cls.btc_df_class.shape[1]) 
        scaler.fit(dummy_data_for_scaler)
        
        model.save(os.path.join(cls.test_models_dir_class, "btc_lstm_best_class.h5"))
        model.save(os.path.join(cls.test_models_dir_class, "aapl_lstm_best_class.h5"))
        joblib.dump(scaler, os.path.join(cls.test_models_dir_class, "btc_scaler_class.joblib"))
        joblib.dump(scaler, os.path.join(cls.test_models_dir_class, "aapl_scaler_class.joblib"))

    @classmethod
    def _mocked_load_data_success_class(cls, ticker, start_date, end_date, progress_bar=None):
        if ticker == 'BTC-USD':
            return pd.read_csv(os.path.join(cls.test_processed_dir_class, "btc_processed_class.csv"), index_col=0, parse_dates=True)
        elif ticker == 'AAPL':
            return pd.read_csv(os.path.join(cls.test_processed_dir_class, "aapl_processed_class.csv"), index_col=0, parse_dates=True)
        elif ticker == 'PETR4.SA': # Adicionado para cobrir opções da UI
            print(f"Aviso: Usando dados mockados de BTC para {ticker} em _mocked_load_data_success_class")
            return pd.read_csv(os.path.join(cls.test_processed_dir_class, "btc_processed_class.csv"), index_col=0, parse_dates=True)
        else:
            raise ValueError(f"Mock Class: Ticker de teste não suportado: {ticker}")

    def setUp(self):
        """Configuração inicial para cada teste (se necessário)."""
        # Resetar mocks do streamlit para cada teste de AppTest para garantir isolamento
        # st.reset_mock() # AppTest lida com seu próprio estado do Streamlit.
        # Se houver mocks específicos por teste, eles podem ser configurados aqui.
        pass
    
    def tearDown(self):
        """Limpeza após cada teste (se necessário)."""
        # Se mocks específicos por teste foram iniciados em setUp, pará-los aqui.
        pass
    
    # O método create_test_data original foi movido e adaptado para _create_test_data_frames_class e _create_mock_models_scalers_class
    # Os testes individuais de data_to_preprocessing_flow e prediction_flow podem precisar de ajustes
    # para usar os dados/modelos de classe ou configurar seus próprios mocks/dados se precisarem de isolamento total.
    # Por ora, eles podem implicitamente usar o patch global de load_data.

    
    def tearDown(self):
        """Limpeza após os testes."""
        # Remover arquivos temporários
        for file in os.listdir(self.test_processed_dir):
            os.remove(os.path.join(self.test_processed_dir, file))
        for file in os.listdir(self.test_models_dir):
            os.remove(os.path.join(self.test_models_dir, file))
        
        # Remover diretórios temporários
        os.rmdir(self.test_processed_dir)
        os.rmdir(self.test_data_dir)
        os.rmdir(self.test_models_dir)
    
    def create_test_data(self):
        """Cria dados de teste para os testes de integração."""
        # Criar DataFrame de teste para BTC
        dates = pd.date_range(start="2023-01-01", periods=200)
        btc_df = pd.DataFrame({
            "Date": dates,
            "Open": 20000 + np.random.randn(200).cumsum(),
            "High": 21000 + np.random.randn(200).cumsum(),
            "Low": 19000 + np.random.randn(200).cumsum(),
            "Close": 20500 + np.random.randn(200).cumsum(),
            "Adj Close": 20500 + np.random.randn(200).cumsum(),
            "Volume": np.random.randint(1000000, 5000000, 200)
        })
        btc_df.set_index("Date", inplace=True)
        
        # Criar DataFrame de teste para AAPL
        aapl_df = pd.DataFrame({
            "Date": dates,
            "Open": 150 + np.random.randn(200).cumsum(),
            "High": 155 + np.random.randn(200).cumsum(),
            "Low": 145 + np.random.randn(200).cumsum(),
            "Close": 152 + np.random.randn(200).cumsum(),
            "Adj Close": 152 + np.random.randn(200).cumsum(),
            "Volume": np.random.randint(10000000, 50000000, 200)
        })
        aapl_df.set_index("Date", inplace=True)
        
        # Salvar DataFrames
        btc_df.to_csv(os.path.join(self.test_processed_dir, "btc_processed.csv"))
        aapl_df.to_csv(os.path.join(self.test_processed_dir, "aapl_processed.csv"))
        
        # Criar modelo e scaler mockados
        import tensorflow as tf
        import joblib
        from sklearn.preprocessing import MinMaxScaler
        
        # Modelo simples para teste
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(60, 6)),
            tf.keras.layers.Dense(14)
        ])
        model.compile(optimizer='adam', loss='mse')
        
        # Scaler simples para teste
        scaler = MinMaxScaler()
        scaler.fit(btc_df.values)
        
        # Salvar modelo e scaler
        model.save(os.path.join(self.test_models_dir, "btc_lstm_best.h5"))
        model.save(os.path.join(self.test_models_dir, "aapl_lstm_best.h5"))
        joblib.dump(scaler, os.path.join(self.test_models_dir, "btc_scaler.joblib"))
        joblib.dump(scaler, os.path.join(self.test_models_dir, "aapl_scaler.joblib"))
    
    def test_data_to_preprocessing_flow(self):
        """Testa o fluxo de dados da ingestão ao pré-processamento."""
        # Mockar o caminho dos dados
        with patch('os.path.join') as mock_path_join:
            # Configurar o mock para retornar caminhos de teste
            mock_path_join.side_effect = lambda *args: os.path.join(*args)
            
            # Carregar dados
            btc_df = loader.load_processed_data("btc", self.test_processed_dir)
            
            # Verificar se os dados foram carregados corretamente
            self.assertIsNotNone(btc_df, "DataFrame BTC deveria ser carregado com sucesso")
            self.assertIn("Adj Close", btc_df.columns, "DataFrame BTC deveria ter coluna 'Adj Close'")
            
            # Aplicar engenharia de features
            btc_df_with_features = feature_engineering.add_technical_indicators(btc_df)
            
            # Verificar se os indicadores foram adicionados
            self.assertIn("SMA_50", btc_df_with_features.columns, "DataFrame deveria ter coluna 'SMA_50'")
            self.assertIn("RSI", btc_df_with_features.columns, "DataFrame deveria ter coluna 'RSI'")
            
            # Tratar valores ausentes
            btc_df_clean = scalers_transformers.handle_missing_values(btc_df_with_features)
            
            # Verificar se não há valores ausentes
            self.assertEqual(btc_df_clean.isna().sum().sum(), 0, "Não deveria haver valores ausentes após tratamento")
    
    def test_prediction_flow(self):
        """Testa o fluxo de previsão completo."""
        # Mockar o caminho dos dados e modelos
        with patch('os.path.join') as mock_path_join:
            # Configurar o mock para retornar caminhos de teste
            def mock_join(*args):
                if "models" in args and "btc_lstm_best.h5" in args:
                    return os.path.join(self.test_models_dir, "btc_lstm_best.h5")
                elif "models" in args and "btc_scaler.joblib" in args:
                    return os.path.join(self.test_models_dir, "btc_scaler.joblib")
                elif "processed" in args and "btc_processed.csv" in args:
                    return os.path.join(self.test_processed_dir, "btc_processed.csv")
                else:
                    return os.path.join(*args)
            
            mock_path_join.side_effect = mock_join
            
            # Carregar dados
            btc_df = pd.read_csv(os.path.join(self.test_processed_dir, "btc_processed.csv"), index_col="Date", parse_dates=True)
            
            # Aplicar engenharia de features
            btc_df = feature_engineering.add_technical_indicators(btc_df)
            btc_df = scalers_transformers.handle_missing_values(btc_df)
            
            # Mockar a função de previsão para evitar problemas com o modelo real
            with patch('src.modeling.prediction.generate_forecast') as mock_generate_forecast:
                # Configurar o mock para retornar uma previsão simulada
                mock_prediction = np.random.rand(1, 14)
                mock_generate_forecast.return_value = mock_prediction
                
                # Executar pipeline de previsão
                model_path = os.path.join(self.test_models_dir, "btc_lstm_best.h5")
                scaler_path = os.path.join(self.test_models_dir, "btc_scaler.joblib")
                
                # Carregar modelo e scaler
                model = prediction.load_model(model_path)
                scaler = prediction.load_scaler(scaler_path)
                
                # Verificar se modelo e scaler foram carregados
                self.assertIsNotNone(model, "Modelo deveria ser carregado com sucesso")
                self.assertIsNotNone(scaler, "Scaler deveria ser carregado com sucesso")
                
                # Preparar sequência de entrada
                input_sequence = prediction.prepare_input_sequence(btc_df, scaler)
                
                # Verificar se a sequência foi preparada corretamente
                self.assertIsNotNone(input_sequence, "Sequência de entrada deveria ser criada com sucesso")
                
                # Gerar previsão
                forecast_scaled = prediction.generate_forecast(model, input_sequence)
                
                # Verificar se a previsão foi gerada
                self.assertIsNotNone(forecast_scaled, "Previsão escalonada deveria ser gerada com sucesso")
                
                # Desnormalizar previsão
                forecast = prediction.inverse_transform_forecast(forecast_scaled, scaler)
                
                # Verificar se a previsão foi desnormalizada
                self.assertIsNotNone(forecast, "Previsão desnormalizada deveria ser gerada com sucesso")
                self.assertEqual(len(forecast), 14, "Previsão deveria ter 14 valores (horizonte de previsão)")
    
    def test_streamlit_ui_components(self):
        """Testa componentes básicos da UI e navegação."""
        # Este teste pode ser expandido ou dividido em testes mais específicos
        pass

    def test_prediction_page_btc(self):
        """Testa a página de previsão de preços para BTC (01_Previsão_de_Preços.py)."""
        at = AppTest.from_file(PREDICTION_PAGE_SCRIPT, default_timeout=30).run()

        # 6.5.2.1. Simular seleção de BTC e um modelo.
        # A página já deve iniciar com BTC selecionado por padrão ou o primeiro da lista.
        # Vamos verificar se o seletor de ativo existe e se BTC é uma opção.
        self.assertIn("asset_selectbox", [w.key for w in at.selectbox])
        asset_selector = at.selectbox(key="asset_selectbox")
        self.assertIn("Bitcoin (BTC-USD)", asset_selector.options)
        asset_selector.select("Bitcoin (BTC-USD)").run()
        
        # Verificar se o seletor de modelo existe
        self.assertIn("model_selectbox", [w.key for w in at.selectbox])
        model_selector = at.selectbox(key="model_selectbox")
        # Selecionar o primeiro modelo disponível (ex: LSTM)
        # O nome exato do modelo pode variar, pegamos o primeiro da lista
        if model_selector.options:
            selected_model_name = model_selector.options[0]
            model_selector.select(selected_model_name).run()
        else:
            self.fail("Nenhum modelo disponível para seleção na página de previsão.")

        # 6.5.2.2. Verificar se o título da página e os elementos chave são renderizados.
        self.assertIn("📈 Previsão de Preços de Ativos", at.title[0].value)
        self.assertTrue(any("Previsão para Bitcoin (BTC-USD)" in h.value for h in at.subheader))
        
        # Verificar se há gráficos (Plotly charts)
        # 6.5.2.3. Verificar se o gráfico de previsão é carregado (presença do elemento).
        # AppTest não interage diretamente com a renderização de gráficos Plotly da mesma forma que um navegador.
        # Podemos verificar se st.plotly_chart foi chamado ou se há elementos que indicam a presença de um gráfico.
        # Por enquanto, vamos verificar se a seção de gráficos existe.
        self.assertTrue(any("Visualização da Previsão vs. Dados Históricos" in h.value for h in at.markdown))

        # 6.5.2.4. Verificar se as previsões exibidas são consistentes
        # Mockar generate_forecast para retornar valores previsíveis
        with patch('src.modeling.prediction.generate_forecast') as mock_generate_forecast,
             patch('src.modeling.prediction.inverse_transform_forecast') as mock_inverse_transform_forecast:
            
            # Configurar o mock para retornar uma previsão simulada e desnormalizada
            # A previsão real tem shape (1, 14), o inverse_transform_forecast retorna (14,)
            mock_forecast_values = np.array([100 + i for i in range(14)]) 
            mock_generate_forecast.return_value = np.random.rand(1, 14) # O valor exato do escalado não importa tanto aqui
            mock_inverse_transform_forecast.return_value = mock_forecast_values
            
            # Re-executar o AppTest ou partes dele se necessário para aplicar o mock
            # Para este caso, como o select já foi feito, o run() pode ser suficiente
            # ou pode ser necessário re-selecionar para triggar a lógica com o mock.
            # Vamos assumir que o select anterior já disparou a lógica e o mock será pego
            # em uma nova renderização implícita ou se chamarmos run() novamente.
            at.run()

            # Verificar se o gráfico de previsão é carregado (presença do elemento plotly_chart)
            self.assertGreater(len(at.plotly_chart), 0, "Deveria haver pelo menos um gráfico Plotly (previsão).")

            # Verificar se a tabela de previsões é exibida e contém valores esperados (ou próximos)
            # A forma como os dados são exibidos (DataFrame, texto) pode variar.
            # Vamos procurar por um DataFrame que possa conter as previsões.
            # Esta verificação é um exemplo e pode precisar de ajuste fino.
            prediction_table_found_and_verified_btc = False
            for df_element in at.dataframe:
                df_data = df_element.value
                if isinstance(df_data, pd.DataFrame) and "Previsão" in df_data.columns:
                    try:
                        displayed_predictions = df_data["Previsão"].values.astype(float)
                        np.testing.assert_array_almost_equal(displayed_predictions, mock_forecast_values, decimal=5)
                        prediction_table_found_and_verified_btc = True
                        break
                    except (ValueError, AssertionError, AttributeError) as e:
                        # AttributeError pode ocorrer se .values não for o esperado ou .astype falhar
                        print(f"DEBUG: Erro ao verificar tabela de previsão BTC: {e}")
                        print(f"DEBUG: Valores exibidos (tipo: {type(df_data['Previsão'].iloc[0] if not df_data.empty else None)}): {df_data['Previsão'].values if not df_data.empty else 'DataFrame vazio'}")
                        print(f"DEBUG: Valores esperados: {mock_forecast_values}")
                        # Continuar procurando se houver múltiplas tabelas ou se a primeira não for a correta
            self.assertTrue(prediction_table_found_and_verified_btc, "Tabela de previsões BTC não encontrada ou valores mockados inconsistentes.")

            # Verificar se a tabela de métricas de desempenho está presente
            self.assertTrue(any("Métricas de Desempenho da Previsão" in h.value for h in at.markdown))
            # Poderíamos também verificar se st.metric foi chamado se usarmos isso para métricas

            # 6.5.5.2 Testar interatividade do slider de dias de previsão
            self.assertTrue(at.slider(key="forecast_days_slider").exists, "Slider de dias de previsão não encontrado.")
            forecast_slider = at.slider(key="forecast_days_slider")
            initial_forecast_days = forecast_slider.value
            forecast_slider.set_value(7) # Mudar para 7 dias
            at.run() # Re-executar para aplicar a mudança no slider

            # Clicar no botão de previsão novamente após mudar o slider
            self.assertTrue(at.button(key="predict_button").exists, "Botão de previsão não encontrado.")
            at.button(key="predict_button").click().run()

            # Verificar se a previsão foi atualizada para 7 dias
            # (Requer que o mock de make_prediction ou a lógica real use o valor do slider)
            # E que a tabela/gráfico reflita isso. 
            # Esta parte pode precisar de mocks mais detalhados ou asserções sobre o output.
            # Por ora, verificamos se o slider mudou e o botão foi clicado.
            self.assertEqual(at.slider(key="forecast_days_slider").value, 7, "Valor do slider de dias não foi atualizado para 7.")

            # Verificar se a tabela de previsões agora tem 7 dias
            # Re-verificar a tabela de previsões após mudar o slider e clicar em prever
            # O mock_forecast_values precisa ser ajustado para 7 dias para esta parte do teste
            mock_forecast_values_7_days = np.array([100 + i for i in range(7)])
            mock_inverse_transform_forecast.return_value = mock_forecast_values_7_days
            at.run() # Re-run para pegar o novo mock na renderização da tabela

            prediction_table_7_days_found = False
            for df_element in at.dataframe:
                df_data = df_element.value
                if isinstance(df_data, pd.DataFrame) and "Previsão" in df_data.columns:
                    if len(df_data) == 7:
                        displayed_predictions_7_days = df_data["Previsão"].values.astype(float)
                        np.testing.assert_array_almost_equal(displayed_predictions_7_days, mock_forecast_values_7_days, decimal=5)
                        prediction_table_7_days_found = True
                        break
            self.assertTrue(prediction_table_7_days_found, "Tabela de previsões com 7 dias não encontrada ou valores inconsistentes.")

            # 6.5.5.2 Testar interatividade do multiselect de indicadores técnicos
            self.assertTrue(at.multiselect(key="indicator_multiselect").exists, "Multiselect de indicadores não encontrado.")
            indicator_multiselect = at.multiselect(key="indicator_multiselect")
            initial_selected_indicators = indicator_multiselect.value
            
            # Selecionar um novo conjunto de indicadores (ex: apenas 'SMA_200' se disponível, ou um default)
            available_indicators = indicator_multiselect.options
            new_selection = []
            if "SMA_200" in available_indicators:
                new_selection = ["SMA_200"]
            elif available_indicators: # Selecionar o primeiro se SMA_200 não estiver lá
                new_selection = [available_indicators[0]]

            if new_selection: # Só prosseguir se houver algo para selecionar
                indicator_multiselect.select(new_selection).run()
                self.assertEqual(at.multiselect(key="indicator_multiselect").value, new_selection, "Seleção do multiselect de indicadores não foi atualizada.")
                # Verificar se o gráfico de indicadores é atualizado (pode ser pela contagem de gráficos ou título)
                # Esta parte é mais complexa de verificar sem acesso direto aos plots gerados.
                # Uma verificação simples é que o app não quebrou e o valor do widget mudou.
                # Poderíamos verificar se um novo st.plotly_chart foi chamado se tivéssemos mocks para isso.

    def test_prediction_page_aapl(self):
        """Testa a página de previsão de preços para AAPL (01_Previsão_de_Preços.py)."""
        at = AppTest.from_file(PREDICTION_PAGE_SCRIPT, default_timeout=30).run()

        # Simular seleção de AAPL
        self.assertIn("asset_selectbox", [w.key for w in at.selectbox])
        asset_selector = at.selectbox(key="asset_selectbox")
        self.assertIn("Apple (AAPL)", asset_selector.options)
        asset_selector.select("Apple (AAPL)").run()
        
        # Selecionar um modelo (o primeiro disponível)
        self.assertIn("model_selectbox", [w.key for w in at.selectbox])
        model_selector = at.selectbox(key="model_selectbox")
        if model_selector.options:
            selected_model_name = model_selector.options[0]
            model_selector.select(selected_model_name).run()
        else:
            self.fail("Nenhum modelo disponível para seleção na página de previsão para AAPL.")

        # Verificar título e elementos chave para AAPL
        self.assertIn("📈 Previsão de Preços de Ativos", at.title[0].value)
        self.assertTrue(any("Previsão para Apple (AAPL)" in h.value for h in at.subheader))
        self.assertTrue(any("Visualização da Previsão vs. Dados Históricos" in h.value for h in at.markdown))

        # Mockar generate_forecast para retornar valores previsíveis
        with patch('src.modeling.prediction.generate_forecast') as mock_generate_forecast,
             patch('src.modeling.prediction.inverse_transform_forecast') as mock_inverse_transform_forecast:
            
            mock_forecast_values_aapl = np.array([150 + i for i in range(14)]) 
            mock_generate_forecast.return_value = np.random.rand(1, 14) 
            mock_inverse_transform_forecast.return_value = mock_forecast_values_aapl
            
            at.run()

            self.assertGreater(len(at.plotly_chart), 0, "Deveria haver pelo menos um gráfico Plotly (previsão AAPL).")

            prediction_table_found_and_verified_aapl = False
            for df_element in at.dataframe:
                df_data = df_element.value
                if isinstance(df_data, pd.DataFrame) and "Previsão" in df_data.columns:
                    try:
                        displayed_predictions_aapl = df_data["Previsão"].values.astype(float)
                        np.testing.assert_array_almost_equal(displayed_predictions_aapl, mock_forecast_values_aapl, decimal=5)
                        prediction_table_found_and_verified_aapl = True
                        break
                    except (ValueError, AssertionError, AttributeError) as e:
                        print(f"DEBUG: Erro ao verificar tabela de previsão AAPL: {e}")
                        print(f"DEBUG: Valores exibidos (tipo: {type(df_data['Previsão'].iloc[0] if not df_data.empty else None)}): {df_data['Previsão'].values if not df_data.empty else 'DataFrame vazio'}")
                        print(f"DEBUG: Valores esperados: {mock_forecast_values_aapl}")
            self.assertTrue(prediction_table_found_and_verified_aapl, "Tabela de previsões AAPL não encontrada ou valores mockados inconsistentes.")

            self.assertTrue(any("Métricas de Desempenho da Previsão" in h.value for h in at.markdown))
        self.assertTrue(len(at.dataframe) > 0, "Deveria haver pelo menos um DataFrame exibido para AAPL.")
        self.assertTrue(any("Métricas de Desempenho da Previsão" in h.value for h in at.markdown))

    def test_exploratory_analysis_page_btc(self):
        """Testa a página de análise exploratória para BTC (02_Análise_Exploratória.py)."""
        at = AppTest.from_file(EXPLORATORY_PAGE_SCRIPT, default_timeout=30).run()

        # 6.5.3.1. Simular seleção de um ativo (BTC)
        self.assertIn("asset_selectbox_exp", [w.key for w in at.selectbox]) # Supondo key diferente da prediction page
        asset_selector = at.selectbox(key="asset_selectbox_exp")
        self.assertIn("Bitcoin (BTC-USD)", asset_selector.options)
        asset_selector.select("Bitcoin (BTC-USD)").run()

        # Verificar a presença de elementos da UI
        self.assertIn("Análise Exploratória para Bitcoin (BTC-USD)", at.title[0].value)
        # Verificar títulos das seções
        markdown_texts = [md.value for md in at.markdown]
        self.assertTrue(any("1. Série Temporal do Preço de Fechamento" in text for text in markdown_texts))
        self.assertTrue(any("2. Volume de Negociação" in text for text in markdown_texts))
        self.assertTrue(any("3. Distribuição dos Retornos Diários" in text for text in markdown_texts))
        self.assertTrue(any("4. Volatilidade Histórica (30 dias)" in text for text in markdown_texts))
        self.assertTrue(any("5. Indicador de Força Relativa (RSI)" in text for text in markdown_texts))
        self.assertTrue(any("6. Estatísticas Descritivas dos Dados Processados" in text for text in markdown_texts))
        self.assertTrue(any("7. Análise de Sazonalidade e Tendência" in text for text in markdown_texts))
        self.assertTrue(any("8. Análise de Volatilidade com GARCH(1,1)" in text for text in markdown_texts))

        # Verificar gráficos Plotly (verificar se há pelo menos o número esperado de gráficos)
        # Os títulos exatos podem variar ou ser complexos de verificar, então contamos os gráficos.
        self.assertGreaterEqual(len(at.plotly_chart), 7, "Número de gráficos plotly para BTC é menor que o esperado") # Ajustar conforme o número de gráficos na página
        
        # Verificar DataFrame de estatísticas descritivas
        self.assertTrue(len(at.dataframe) > 0, "Nenhum DataFrame de estatísticas encontrado para BTC")
        # Verificar sumário do GARCH (st.text)
        self.assertTrue(any("Sumário do Modelo GARCH(1,1)" in subheader.value for subheader in at.subheader), "Subheader do GARCH não encontrado para BTC")
        self.assertTrue(len(at.text) > 0, "Nenhum texto de sumário GARCH encontrado para BTC")

    def test_exploratory_analysis_page_aapl(self):
        """Testa a página de análise exploratória para AAPL (02_Análise_Exploratória.py)."""
        at = AppTest.from_file(EXPLORATORY_PAGE_SCRIPT, default_timeout=30).run()

        # Simular seleção de AAPL
        self.assertIn("asset_selectbox_exp", [w.key for w in at.selectbox])
        asset_selector = at.selectbox(key="asset_selectbox_exp")
        self.assertIn("Apple (AAPL)", asset_selector.options)
        asset_selector.select("Apple (AAPL)").run()

        # Verificar a presença de elementos da UI
        self.assertIn("Análise Exploratória para Apple (AAPL)", at.title[0].value)
        # Verificar títulos das seções
        markdown_texts = [md.value for md in at.markdown]
        self.assertTrue(any("1. Série Temporal do Preço de Fechamento" in text for text in markdown_texts))
        self.assertTrue(any("2. Volume de Negociação" in text for text in markdown_texts))
        self.assertTrue(any("3. Distribuição dos Retornos Diários" in text for text in markdown_texts))
        self.assertTrue(any("4. Volatilidade Histórica (30 dias)" in text for text in markdown_texts))
        self.assertTrue(any("5. Indicador de Força Relativa (RSI)" in text for text in markdown_texts))
        self.assertTrue(any("6. Estatísticas Descritivas dos Dados Processados" in text for text in markdown_texts))
        self.assertTrue(any("7. Análise de Sazonalidade e Tendência" in text for text in markdown_texts))
        self.assertTrue(any("8. Análise de Volatilidade com GARCH(1,1)" in text for text in markdown_texts))

        # Verificar gráficos Plotly
        self.assertGreaterEqual(len(at.plotly_chart), 7, "Número de gráficos plotly para AAPL é menor que o esperado")

        # Verificar DataFrame de estatísticas descritivas
        self.assertTrue(len(at.dataframe) > 0, "Nenhum DataFrame de estatísticas encontrado para AAPL")
        # Verificar sumário do GARCH (st.text)
        self.assertTrue(any("Sumário do Modelo GARCH(1,1)" in subheader.value for subheader in at.subheader), "Subheader do GARCH não encontrado para AAPL")
        self.assertTrue(len(at.text) > 0, "Nenhum texto de sumário GARCH encontrado para AAPL")

    def test_full_app_flow_navigation(self):
        """Testa a navegação básica entre as páginas da aplicação (Home -> Previsão -> Análise)."""
        # 6.5.4.1. Iniciar na Home Page e verificar elementos.
        at_home = AppTest.from_file(HOME_PAGE_SCRIPT, default_timeout=30).run()
        self.assertIn("Página Inicial", at_home.title[0].value) # Ajustar conforme o título real da Home
        self.assertTrue(any("Bem-vindo ao Sistema de Análise e Previsão de Ativos Financeiros" in m.value for m in at_home.markdown))

        # A navegação com AppTest entre páginas de um app multipage é um pouco diferente.
        # AppTest.from_file() testa um script isoladamente.
        # Para testar a navegação real, precisaríamos simular cliques nos links de navegação
        # que o Streamlit gera, o que pode ser complexo com AppTest diretamente.
        # Uma abordagem é verificar se os scripts das páginas são carregáveis e se os títulos mudam.

        # 6.5.4.2. Navegar para a página de Previsão de Preços e verificar.
        # Como não podemos "clicar" em um link de navegação diretamente para outra página com AppTest,
        # vamos simular carregando o script da página de previsão diretamente.
        # O teste real de navegação do Streamlit multipage é mais e2e.
        at_prediction = AppTest.from_file(PREDICTION_PAGE_SCRIPT, default_timeout=30).run()
        self.assertIn("📈 Previsão de Preços de Ativos", at_prediction.title[0].value)
        # Verificar um elemento específico da página de previsão
        self.assertTrue(at_prediction.selectbox(key="asset_selectbox").exists())

        # 6.5.4.3. Navegar para a página de Análise Exploratória e verificar.
        at_exploratory = AppTest.from_file(EXPLORATORY_PAGE_SCRIPT, default_timeout=30).run()
        self.assertIn("📊 Análise Exploratória de Dados (EDA)", at_exploratory.title[0].value)
        # Verificar um elemento específico da página de análise exploratória
        self.assertTrue(at_exploratory.selectbox(key="asset_selectbox_exp").exists())

        # Este teste verifica se cada página principal pode ser carregada e tem seu título correto.
        # Testes de navegação mais profundos exigiriam ferramentas de teste E2E como Selenium ou Playwright
        # se a interação exata com os widgets de navegação do Streamlit for necessária.

    @patch('src.data_ingestion.loader.load_data')
    def test_error_handling_data_loading_failure_prediction_page(self, mock_load_data):
        """Testa o tratamento de erro na página de previsão quando o carregamento de dados falha."""
        # 6.5.5.1. Simular falha no carregamento de dados (e.g., ativo inválido, API offline).
        mock_load_data.side_effect = Exception("Erro simulado ao carregar dados")

        at = AppTest.from_file(PREDICTION_PAGE_SCRIPT, default_timeout=30).run()

        # Selecionar um ativo qualquer, o mock vai causar a falha
        asset_selector = at.selectbox(key="asset_selectbox")
        if asset_selector.options:
            asset_selector.select(asset_selector.options[0]).run()
        else:
            self.fail("Nenhum ativo disponível para seleção no teste de falha de carregamento.")

        # 6.5.5.2. Verificar se uma mensagem de erro apropriada é exibida para o usuário.
        # A forma como o erro é exibido pode variar (st.error, st.exception, etc.)
        # Vamos procurar por st.error, que é comum para mensagens de erro ao usuário.
        self.assertTrue(len(at.error) > 0, "Deveria haver uma mensagem de erro st.error.")
        self.assertTrue(any("Erro ao carregar dados" in e.value.lower() or "erro simulado ao carregar dados" in e.value.lower() for e in at.error), 
                        f"Mensagem de erro não encontrada ou inesperada. Erros encontrados: {[e.value for e in at.error]}")

    @patch('src.data_ingestion.loader.load_data')
    def test_error_handling_data_loading_failure_exploratory_page(self, mock_load_data):
        """Testa o tratamento de erro na página de análise exploratória quando o carregamento de dados falha."""
        mock_load_data.side_effect = Exception("Erro simulado EDA")

        at = AppTest.from_file(EXPLORATORY_PAGE_SCRIPT, default_timeout=30).run()

        asset_selector = at.selectbox(key="asset_selectbox_exp")
        if asset_selector.options:
            asset_selector.select(asset_selector.options[0]).run()
        else:
            self.fail("Nenhum ativo disponível para seleção no teste de falha de carregamento EDA.")
        
        self.assertTrue(len(at.error) > 0, "Deveria haver uma mensagem de erro st.error na página EDA.")
        self.assertTrue(any("erro ao carregar dados" in e.value.lower() or "erro simulado eda" in e.value.lower() for e in at.error),
                        f"Mensagem de erro EDA não encontrada ou inesperada. Erros: {[e.value for e in at.error]}")

    # Adicionar mais testes de tratamento de erro conforme necessário, 
    # por exemplo, para falhas de modelo, dados insuficientes, etc.


        """Testa a integração dos componentes de UI do Streamlit."""
        # Configurar mocks para componentes do Streamlit
        st.selectbox.side_effect = ["Bitcoin (BTC-USD)", "LSTM"]
        
        # Testar seletores de ativo e modelo
        asset_name, asset_code = ui_elements.create_asset_selector()
        model_name, model_code = ui_elements.create_model_selector()
        
        # Verificar resultados
        self.assertEqual(asset_code, "btc", "Código do ativo deveria ser 'btc'")
        self.assertEqual(model_code, "lstm", "Código do modelo deveria ser 'lstm'")
        
        # Carregar dados
        btc_df = pd.read_csv(os.path.join(self.test_processed_dir, "btc_processed.csv"), index_col="Date", parse_dates=True)
        
        # Aplicar engenharia de features
        btc_df = feature_engineering.add_technical_indicators(btc_df)
        
        # Testar plotagem
        fig_hist = plotting.plot_historical_data(btc_df, asset_name)
        self.assertIsNotNone(fig_hist, "Figura de histórico deveria ser criada com sucesso")
        
        # Testar plotagem de indicadores
        indicators = ["SMA_50", "RSI"]
        fig_ind = plotting.plot_technical_indicators(btc_df, indicators)
        self.assertIsNotNone(fig_ind, "Figura de indicadores deveria ser criada com sucesso")
        
        # Testar plotagem de previsão
        forecast_dates = pd.date_range(start=btc_df.index[-1] + pd.Timedelta(days=1), periods=14)
        forecast_values = btc_df["Adj Close"].iloc[-1] * (1 + np.random.randn(14)*0.02).cumsum()
        
        fig_forecast = plotting.plot_forecast_vs_actual(btc_df, forecast_values, forecast_dates, asset_name, model_name)
        self.assertIsNotNone(fig_forecast, "Figura de previsão deveria ser criada com sucesso")

if __name__ == '__main__':
    unittest.main()

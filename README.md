# MIT-510 - Previsão de Ativos Financeiros com IA

## 📊 Visão Geral

O **MIT-510** é um sistema completo e avançado de previsão de séries temporais financeiras utilizando Redes Neurais Recorrentes (RNNs) e técnicas modernas de Machine Learning. O projeto implementa modelos de aprendizado profundo para prever preços de Bitcoin (BTC-USD) e ações da Apple (AAPL) com horizonte de 14 dias, incluindo análise exploratória avançada, múltiplas estratégias de trading e métricas financeiras sofisticadas.

### 🎯 Objetivos Principais

- Desenvolver um pipeline completo de ciência de dados para previsão financeira
- Implementar e comparar diferentes arquiteturas RNN (SimpleRNN, LSTM, GRU)
- Criar análise exploratória interativa com técnicas estatísticas avançadas
- Desenvolver aplicativo web Streamlit com interface moderna e intuitiva
- Avaliar desempenho através de métricas financeiras e simulação de estratégias
- Implementar indicadores técnicos conforme padrões da indústria
- Integrar variáveis exógenas para melhorar a precisão dos modelos
- Fornecer análises de correlação, cointegração e causalidade de Granger

## 🏗️ Arquitetura do Sistema

### Estrutura de Diretórios

```
MIT-510/
├── data/
│   ├── raw/                    # Dados brutos (Kaggle, CSV)
│   │   ├── BTC-USD From 2014 To Dec-2024.csv
│   │   ├── apple-stockprice-2014-2024.csv
│   │   └── ...
│   └── processed/              # Dados processados
│       ├── btc_processed.csv
│       └── aapl_processed.csv
├── docs/
│   ├── Arquitetura de Projeto IA - Previsão BTC-AAPL.pdf
│   ├── TODO.md                 # Roadmap detalhado do projeto
│   ├── requirements.txt        # Dependências do projeto
│   └── README.md              # Este arquivo
├── models/                     # Modelos treinados e scalers
├── notebooks/
│   └── financial_forecasting_eda_modeling.ipynb  # Notebook principal
├── reports/
│   ├── figures/               # Gráficos e visualizações
│   ├── coverage/              # Relatórios de cobertura de testes
│   ├── relatorio_projeto.md   # Relatório técnico completo
│   └── relatorio_projeto.pdf
├── src/
│   ├── app/
│   │   ├── app.py             # Aplicativo Streamlit principal
│   │   ├── components/        # Componentes reutilizáveis
│   │   └── pages/             # Páginas do aplicativo
│   ├── data_ingestion/
│   │   └── loader.py          # Carregamento e consolidação de dados
│   ├── modeling/
│   │   ├── rnn_models.py      # Definição dos modelos RNN
│   │   ├── prediction.py      # Funções de predição
│   │   └── strategy_simulation.py  # Simulação de estratégias
│   ├── preprocessing/
│   │   ├── feature_engineering.py  # Indicadores técnicos
│   │   └── scalers_transformers.py # Normalização de dados
│   └── utils/
└── tests/
    ├── integration/           # Testes de integração
    ├── unit/                  # Testes unitários
    └── notebook_tests/        # Testes do Jupyter Notebook
```

### 🔄 Fluxo de Dados

1. **Ingestão**: Carregamento de dados históricos (Kaggle) + dados recentes (yfinance)
2. **Consolidação**: Combinação e tratamento de sobreposições
3. **Pré-processamento**: Cálculo de indicadores técnicos (SMA, EMA, RSI, MACD)
4. **Normalização**: Escalonamento para treinamento dos modelos
5. **Modelagem**: Treinamento de RNNs (SimpleRNN, LSTM, GRU)
6. **Avaliação**: Métricas de desempenho e simulação de estratégias
7. **Visualização**: Apresentação via Jupyter Notebook e Streamlit

## 🚀 Configuração do Ambiente

### Pré-requisitos

- Python 3.8+
- pip ou conda
- Git

### 1. Clonagem do Repositório

```bash
git clone <url-do-repositorio>
cd MIT-510
```

### 2. Criação do Ambiente Virtual

```bash
# Usando venv
python -m venv mit-510
# ou
py -m venv mit-510

# Ativação (Windows)
mit-510\Scripts\activate

# Ativação (Linux/Mac)
source mit-510/bin/activate
```

### 3. Instalação das Dependências

```bash
pip install -r docs/requirements.txt
```

### Principais Dependências

- **Dados**: `pandas`, `numpy`, `yfinance`
- **ML/DL**: `tensorflow`, `scikit-learn`
- **Indicadores Técnicos**: `ta`, `TA-Lib`
- **Visualização**: `matplotlib`, `seaborn`, `plotly`
- **Web App**: `streamlit`
- **Testes**: `pytest`, `pytest-cov`, `nbval`
- **Análise**: `statsmodels`

## 📊 Execução do Projeto

### 1. Jupyter Notebook (Análise Completa)

```bash
# Iniciar Jupyter
jupyter notebook

# Abrir o notebook principal
# notebooks/financial_forecasting_eda_modeling.ipynb
```

**O notebook inclui:**

- Análise Exploratória de Dados (EDA)
- Engenharia de Features
- Implementação e treinamento de modelos RNN
- Avaliação de desempenho
- Simulação de estratégias de trading
- Visualizações interativas

### 2. Aplicativo Streamlit (Interface Web)

```bash
# Executar o aplicativo
streamlit run src/app/app.py

# Ou a partir da raiz do projeto
cd src/app
streamlit run app.py
```

**Páginas do Aplicativo:**

1. **📈 Previsão de Preços** (`01_Previsão_de_Preços.py`)
   - Seleção de ativos (BTC, AAPL) e modelos (LSTM, GRU, SimpleRNN)
   - Visualização de previsões de 14 dias
   - Gráficos interativos com Plotly
   - Indicadores técnicos em tempo real

2. **📊 Análise Exploratória** (`02_Análise_Exploratória.py`)
   - Matriz de correlação interativa
   - Análise de cointegração (Teste de Engle-Granger)
   - Causalidade de Granger bidirecional
   - Modelagem GARCH para volatilidade
   - Análise de clusters de volatilidade
   - Decomposição sazonal e tendências

3. **📈 Estratégias de Trading** (`03_Estratégias_de_Trading.py`)
   - Simulação de múltiplas estratégias
   - Comparação automática de performance
   - Gestão de risco com stop-loss/take-profit
   - Visualização de curvas de capital

4. **📊 Métricas do Modelo** (`04_Métricas_do_Modelo.py`)
   - Métricas financeiras avançadas
   - Comparação entre modelos
   - Análise de performance detalhada
   - Visualizações de erro e precisão

5. **📚 Documentação** (`05_Documentação.py`)
   - Documentação completa do projeto
   - Guias de uso e configuração
   - Informações técnicas detalhadas

### 3. Execução de Testes

```bash
# Todos os testes
pytest

# Testes com cobertura
pytest --cov=src --cov-report=html

# Teste do notebook
pytest --nbval notebooks/financial_forecasting_eda_modeling.ipynb

# Testes específicos
pytest tests/unit/
pytest tests/integration/
```

## 🤖 Modelos Implementados

### Arquiteturas RNN

1. **SimpleRNN**

   - Rede neural recorrente básica
   - Boa para padrões simples
   - Menor complexidade computacional
2. **LSTM (Long Short-Term Memory)**

   - Resolve problema do gradiente que desaparece
   - Captura dependências de longo prazo
   - Melhor para séries complexas
3. **GRU (Gated Recurrent Unit)**

   - Versão simplificada do LSTM
   - Menos parâmetros, treinamento mais rápido
   - Performance similar ao LSTM

### Configurações dos Modelos

- **Janela de entrada**: 60 dias
- **Horizonte de previsão**: 14 dias
- **Features**: Preços (OHLC) + Indicadores técnicos
- **Normalização**: MinMaxScaler
- **Otimizador**: Adam
- **Função de perda**: MSE
- **Callbacks**: EarlyStopping, ReduceLROnPlateau

## 📈 Indicadores Técnicos Implementados

### Indicadores Principais (Conforme Documento de Qualidade)

- **Bollinger Bands**: Bandas de volatilidade baseadas em desvio padrão
- **Stochastic Oscillator**: Oscilador estocástico para momentum
- **Williams %R**: Indicador de momentum e sobrecompra/sobrevenda
- **CCI (Commodity Channel Index)**: Índice de canal de commodities
- **ADX (Average Directional Index)**: Força da tendência
- **OBV (On-Balance Volume)**: Volume em equilíbrio
- **ATR (Average True Range)**: Volatilidade média verdadeira
- **Momentum**: Indicador de momentum de preços
- **ROC (Rate of Change)**: Taxa de mudança
- **TRIX**: Indicador de momentum suavizado

### Indicadores Legados (Opcionais)

- **SMA (Simple Moving Average)**: Médias móveis simples
- **EMA (Exponential Moving Average)**: Médias móveis exponenciais
- **RSI (Relative Strength Index)**: Índice de força relativa
- **MACD (Moving Average Convergence Divergence)**: Convergência/divergência de médias

## 📊 Métricas de Avaliação Avançadas

### Métricas Estatísticas

- **RMSE (Root Mean Square Error)**: Erro quadrático médio
- **MAE (Mean Absolute Error)**: Erro absoluto médio
- **MAPE (Mean Absolute Percentage Error)**: Erro percentual absoluto médio
- **R² Score**: Coeficiente de determinação

### Métricas Financeiras Avançadas

- **Sharpe Ratio**: Relação risco-retorno ajustada
- **Sortino Ratio**: Sharpe ratio considerando apenas downside risk
- **Information Ratio**: Retorno ativo dividido pelo tracking error
- **Calmar Ratio**: Retorno anualizado dividido pelo maximum drawdown
- **Maximum Drawdown**: Maior perda consecutiva do pico ao vale
- **Acurácia Direcional**: Porcentagem de previsões corretas de direção
- **Retorno Total**: Performance absoluta da estratégia
- **Volatilidade Anualizada**: Desvio padrão dos retornos anualizados

## 🎯 Estratégias de Trading Avançadas

### Estratégias Implementadas

1. **Buy & Hold**: Estratégia passiva de comprar e manter
2. **Momentum Strategy**: Baseada em sinais de momentum dos indicadores
3. **Momentum + Risk Management**: Momentum com stop-loss e take-profit
4. **Mean Reversion**: Estratégia de reversão à média
5. **Mean Reversion + Risk Management**: Reversão à média com gestão de risco

### Características das Estratégias

- **Gestão de Risco**: Stop-loss (5%) e take-profit (10%) implementados
- **Sinais Técnicos**: Baseados em múltiplos indicadores técnicos
- **Rebalanceamento**: Dinâmico baseado em previsões e sinais
- **Comparação Automática**: Sistema de ranking de performance

### Simulação Avançada

- Capital inicial: $10,000
- Gestão de posições com stop-loss e take-profit
- Log detalhado de operações
- Análise de drawdown e períodos de recuperação

## 🧪 Testes e Qualidade

### Estrutura de Testes

- **Testes Unitários**: Funções individuais
- **Testes de Integração**: Fluxo completo
- **Testes de Notebook**: Validação do Jupyter
- **Cobertura de Código**: Relatórios HTML

### Execução

```bash
# Executar todos os testes
pytest

# Com relatório de cobertura
pytest --cov=src --cov-report=html

# Ver relatório
open reports/coverage/index.html
```

## 📋 Roadmap e Status

Consulte o arquivo `docs/TODO.md` para:

- Lista detalhada de tarefas
- Status de implementação
- Próximos passos
- Estrutura de desenvolvimento em fases

## 🔧 Desenvolvimento e Extensibilidade

### Adicionando Novos Ativos

1. Adicionar dados em `data/raw/`
2. Atualizar `loader.py` para novo ativo
3. Modificar `feature_engineering.py` para novos indicadores
4. Criar página no Streamlit
5. Adicionar testes correspondentes
6. Treinar modelos específicos

### Novos Modelos

1. Implementar em `modeling/rnn_models.py`
2. Adicionar ao script de treinamento `train_enhanced_models.py`
3. Integrar ao aplicativo Streamlit
4. Criar testes unitários
5. Adicionar métricas de avaliação

### Novas Estratégias

1. Implementar em `modeling/strategy_simulation.py`
2. Adicionar à página de estratégias
3. Incluir no sistema de comparação
4. Documentar parâmetros e lógica

### Variáveis Exógenas

1. Implementar coleta em `data_ingestion/exogenous_data.py`
2. Integrar ao pré-processamento
3. Modificar arquiteturas dos modelos
4. Atualizar pipeline de treinamento

## 📚 Documentação Adicional

- **Relatório Técnico**: `reports/relatorio_projeto.md`
- **Relatório de Treinamento**: `reports/enhanced_model_training_report.md`
- **Arquitetura Detalhada**: `docs/Arquitetura de Projeto IA - Previsão BTC-AAPL.pdf`
- **Roadmap Completo**: `docs/TODO.md`
- **Cobertura de Testes**: `reports/coverage/index.html`
- **Documentação de Conformidade**: Documento de qualidade integrado
- **Jupyter Notebook**: `notebooks/financial_forecasting_eda_modeling.ipynb`

## 🏆 Status do Projeto

### Funcionalidades Implementadas ✅

- ✅ Pipeline completo de dados (ingestão, processamento, normalização)
- ✅ 10 indicadores técnicos conforme padrões da indústria
- ✅ 3 arquiteturas RNN (SimpleRNN, LSTM, GRU)
- ✅ Métricas financeiras avançadas (Sharpe, Sortino, Information Ratio, etc.)
- ✅ 5 estratégias de trading com gestão de risco
- ✅ Análise exploratória avançada (correlação, cointegração, GARCH)
- ✅ Aplicativo Streamlit com 5 páginas funcionais
- ✅ Sistema de testes unitários e de integração
- ✅ Otimização de hiperparâmetros com Optuna
- ✅ Modelos básicos e avançados treinados

### Em Desenvolvimento 🚧

- 🚧 Implementação completa de variáveis exógenas
- 🚧 Relatório final para conformidade acadêmica
- 🚧 Validação final de conformidade

### Próximos Passos 📋

- 📋 Integração de dados macroeconômicos
- 📋 Expansão para outros ativos financeiros
- 📋 Implementação de modelos ensemble
- 📋 Sistema de alertas em tempo real

## ⚠️ Limitações e Considerações

### Limitações Técnicas

- Horizonte de previsão limitado a 14 dias
- Modelos baseados apenas em dados históricos
- Simulação sem custos de transação e slippage
- Não considera eventos de mercado extraordinários
- Variáveis exógenas limitadas aos dados disponíveis
- Overfitting pode ocorrer em períodos de alta volatilidade

### Considerações Estatísticas

- Séries temporais financeiras são não-estacionárias
- Presença de heterocedasticidade (volatilidade variável)
- Possível autocorrelação nos resíduos
- Quebras estruturais podem afetar a performance
- Regime changes não são explicitamente modelados

### Uso Responsável

- **NÃO** deve ser usado para decisões reais de investimento
- Projeto educacional e de demonstração técnica
- Mercados financeiros são imprevisíveis e complexos
- Sempre consulte profissionais qualificados
- Performance passada não garante resultados futuros
- Considere sempre a gestão de risco adequada

## 🤝 Contribuição

1. Fork do repositório
2. Criar branch para feature (`git checkout -b feature/nova-funcionalidade`)
3. Commit das mudanças (`git commit -am 'Adiciona nova funcionalidade'`)
4. Push para branch (`git push origin feature/nova-funcionalidade`)
5. Criar Pull Request

## 📄 Licença

Este projeto é desenvolvido para fins educacionais. Consulte o arquivo LICENSE para detalhes.

## 📞 Contato

Para dúvidas ou sugestões sobre o projeto, envie e-mail para o autor adiliobb@gmail.com.

---

## 🎓 Contexto Acadêmico

Este projeto foi desenvolvido como parte do curso MIT-510 e representa uma implementação completa de um sistema de previsão financeira usando técnicas modernas de Machine Learning e Deep Learning. O projeto demonstra:

- Aplicação prática de Redes Neurais Recorrentes em finanças
- Implementação de pipeline completo de ciência de dados
- Desenvolvimento de aplicação web interativa
- Análise estatística avançada de séries temporais
- Avaliação rigorosa de modelos com métricas financeiras
- Simulação de estratégias de trading quantitativo

**Disclaimer**: Este projeto é exclusivamente para fins educacionais e de pesquisa acadêmica. Não constitui aconselhamento financeiro e não deve ser usado para decisões reais de investimento. Os resultados apresentados são baseados em dados históricos e não garantem performance futura.

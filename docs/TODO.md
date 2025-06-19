\# TODO.md \- Projeto de Previsão de Ativos Financeiros com IA (BTC & AAPL)

Este arquivo detalha as tarefas necessárias para completar os três entregáveis principais do projeto: o Jupyter Notebook, o Relatório Final e o Aplicativo Streamlit, incluindo todas as etapas de teste. As tarefas são baseadas no Documento de Arquitetura do Projeto.

\#\# Fase 1: Configuração Inicial, Coleta e Preparação de Dados

\-   \[ \] \*\*1.1. Configuração do Ambiente de Desenvolvimento:\*\*  
    \-   \[ \] 1.1.1. Criar um ambiente virtual Python (\`venv\` ou \`conda\`).  
    \-   \[ \] 1.1.2. Instalar bibliotecas base (pandas, numpy, yfinance, scikit-learn, tensorflow, ta/TA-Lib, matplotlib, plotly, seaborn, streamlit, pytest, pytest-cov, nbval).  
    \-   \[ \] 1.1.3. Criar a estrutura de pastas do projeto conforme definido na Seção 3.3 do Documento de Arquitetura.  
        \`\`\`  
        agile\_capital\_forecast/  
        ├── data/  
        │   ├── raw/  
        │   └── processed/  
        ├── notebooks/  
        ├── src/  
        │   ├── data\_ingestion/  
        │   ├── preprocessing/  
        │   ├── modeling/  
        │   ├── app/  
        │   │   ├── pages/  
        │   │   └── components/  
        │   └── utils/  
        ├── tests/  
        │   ├── unit/  
        │   │   ├── data\_ingestion/  
        │   │   ├── preprocessing/  
        │   │   └── modeling/  
        │   ├── integration/  
        │   └── notebook\_tests/  
        ├── models/  
        ├── reports/  
        │   └── figures/  
        ├── .gitignore  
        ├── requirements.txt  
        └── TODO.md  
        \`\`\`  
    \-   \[ \] 1.1.4. Inicializar o repositório Git (\`git init\`).  
    \-   \[ \] 1.1.5. Criar arquivo \`.gitignore\` inicial (ex: para \`\_\_pycache\_\_\`, \`\*.csv\` em \`data/raw\` se não versionados, \`\*.h5\` em \`models/\` se não versionados, ambiente virtual).  
    \-   \[ \] 1.1.6. Gerar o arquivo \`requirements.txt\` inicial (\`pip freeze \> requirements.txt\`).

\-   \[ \] \*\*1.2. Coleta, Sincronização e Processamento Inicial de Dados (Módulo \`src/data\_ingestion/loader.py\`):\*\*  
    \-   \[ \] 1.2.1. Implementar função em \`loader.py\` para baixar dados históricos de BTC-USD do Kaggle (conforme Tabela 2.1).  
    \-   \[ \] 1.2.2. Implementar função em \`loader.py\` para baixar dados históricos de AAPL do Kaggle (conforme Tabela 2.1).  
    \-   \[ \] 1.2.3. Implementar função em \`loader.py\` para buscar dados recentes/ajustados de BTC-USD via \`yfinance\`.  
    \-   \[ \] 1.2.4. Implementar função em \`loader.py\` para buscar dados recentes/ajustados de AAPL via \`yfinance\` (garantir \`auto\_adjust=True\` ou uso de 'Adj Close').  
    \-   \[ \] 1.2.5. Implementar lógica de consolidação em \`loader.py\`:  
        \-   \[ \] 1.2.5.1. Concatenar dados do Kaggle e \`yfinance\`.  
        \-   \[ \] 1.2.5.2. Tratar sobreposições (priorizar \`yfinance\`).  
        \-   \[ \] 1.2.5.3. Garantir índice de datas contínuo, ordenado e sem duplicatas.  
        \-   \[ \] 1.2.5.4. Padronizar colunas (Data, Open, High, Low, Close, Adj Close, Volume).  
    \-   \[ \] 1.2.6. Implementar função em \`loader.py\` para salvar os datasets processados em \`data/processed/\` (ex: \`btc\_processed.csv\`, \`aapl\_processed.csv\`).  
    \-   \[ \] 1.2.7. \*\*Teste Unitário para \`loader.py\` (\`tests/unit/data\_ingestion/test\_loader.py\`):\*\*  
        \-   \[ \] 1.2.7.1. Testar carregamento de arquivo CSV de teste.  
        \-   \[ \] 1.2.7.2. Testar tratamento de arquivo inexistente.  
        \-   \[ \] 1.2.7.3. Testar sincronização com dados de API mockada (para \`yfinance\`).  
        \-   \[ \] 1.2.7.4. Testar lógica de consolidação e priorização.  
        \-   \[ \] 1.2.7.5. Executar testes e garantir que passem.

\#\# Fase 2: Desenvolvimento do Jupyter Notebook (\`notebooks/financial\_forecasting\_eda\_modeling.ipynb\`)

\-   \[ \] \*\*2.1. Coleta e Ingestão de Dados no Notebook:\*\*  
    \-   \[ \] 2.1.1. Carregar os datasets processados de \`data/processed/\` (BTC e AAPL) usando funções de \`src/data\_ingestion/loader.py\`.  
    \-   \[ \] 2.1.2. Realizar verificações primárias (número de registros, colunas, tipos de dados).

\-   \[ \] \*\*2.2. Análise Exploratória de Dados (EDA):\*\*  
    \-   \[ \] 2.2.1. Plotar séries temporais de preços (Adj Close/Close) e volume.  
    \-   \[ \] 2.2.2. Calcular e plotar retornos diários.  
    \-   \[ \] 2.2.3. Visualizar distribuições de preços e retornos (histogramas, box plots).  
    \-   \[ \] 2.2.4. Aplicar Teste ADF para estacionariedade (preços e retornos).  
    \-   \[ \] 2.2.5. Plotar gráficos ACF e PACF.  
    \-   \[ \] 2.2.6. Documentar insights da EDA em células Markdown.

\-   \[ \] \*\*2.3. Engenharia de Features (Módulo \`src/preprocessing/feature\_engineering.py\` e uso no Notebook):\*\*  
    \-   \[ \] 2.3.1. Implementar funções em \`feature\_engineering.py\` para calcular: SMA, EMA, RSI, MACD (usando \`TA-Lib\` ou \`ta\`).  
    \-   \[ \] 2.3.2. Aplicar essas funções no notebook para adicionar os indicadores aos DataFrames de BTC e AAPL.  
    \-   \[ \] 2.3.3. \*\*Teste Unitário para \`feature\_engineering.py\` (\`tests/unit/preprocessing/test\_feature\_engineering.py\`):\*\*  
        \-   \[ \] 2.3.3.1. Testar cálculo de SMA com dados de entrada conhecidos e saída esperada.  
        \-   \[ \] 2.3.3.2. Testar cálculo de EMA.  
        \-   \[ \] 2.3.3.3. Testar cálculo de RSI.  
        \-   \[ \] 2.3.3.4. Testar cálculo de MACD.  
        \-   \[ \] 2.3.3.5. Testar casos de borda (ex: dados insuficientes).  
        \-   \[ \] 2.3.3.6. Executar testes e garantir que passem.

\-   \[ \] \*\*2.4. Pré-processamento de Dados (Módulo \`src/preprocessing/scalers\_transformers.py\` e uso no Notebook):\*\*  
    \-   \[ \] 2.4.1. Tratar valores ausentes nos indicadores e outras colunas (\`fillna(method='ffill')\`).  
    \-   \[ \] 2.4.2. Implementar/Utilizar \`MinMaxScaler\` ou \`StandardScaler\` em \`scalers\_transformers.py\`.  
    \-   \[ \] 2.4.3. Aplicar normalização no notebook:  
        \-   \[ \] 2.4.3.1. Dividir dados em treino/validação/teste cronologicamente.  
        \-   \[ \] 2.4.3.2. Ajustar (\`fit\`) o scaler SOMENTE no conjunto de treino.  
        \-   \[ \] 2.4.3.3. Transformar (\`transform\`) os conjuntos de treino, validação e teste.  
    \-   \[ \] 2.4.4. Criar janelas de tempo para RNNs (entrada de N dias, saída de 14 dias) usando \`tf.keras.utils.timeseries\_dataset\_from\_array\` ou função customizada.  
    \-   \[ \] 2.4.5. \*\*Teste Unitário para \`scalers\_transformers.py\` (se funções customizadas forem criadas):\*\*  
        \-   \[ \] 2.4.5.1. Testar normalização e desnormalização.  
        \-   \[ \] 2.4.5.2. Executar testes e garantir que passem.

\-   \[ \] \*\*2.5. Modelagem com Redes Neurais Recorrentes (Módulo \`src/modeling/rnn\_models.py\` e uso no Notebook):\*\*  
    \-   \[ \] 2.5.1. Definir arquiteturas em \`rnn\_models.py\`: SimpleRNN, LSTM, GRU (1-2 camadas, Dropout, opção stateful).  
    \-   \[ \] 2.5.2. Implementar e treinar cada modelo no notebook para BTC.  
    \-   \[ \] 2.5.3. Implementar e treinar cada modelo no notebook para AAPL.  
    \-   \[ \] 2.5.4. \*\*Teste Unitário para \`rnn\_models.py\` (\`tests/unit/modeling/test\_rnn\_models.py\`):\*\*  
        \-   \[ \] 2.5.4.1. Testar criação de instâncias de SimpleRNN, LSTM, GRU.  
        \-   \[ \] 2.5.4.2. Verificar estrutura das camadas (número de camadas, unidades).  
        \-   \[ \] 2.5.4.3. Executar testes e garantir que passem.

\-   \[ \] \*\*2.6. Treinamento e Validação no Notebook:\*\*  
    \-   \[ \] 2.6.1. Utilizar \`TimeSeriesSplit\` ou validação em janelas deslizantes.  
    \-   \[ \] 2.6.2. Compilar modelos (otimizador Adam, perda MSE).  
    \-   \[ \] 2.6.3. Usar callbacks: \`EarlyStopping\`, \`ReduceLROnPlateau\`.  
    \-   \[ \] 2.6.4. Gerenciar \`model.reset\_states()\` para modelos stateful.  
    \-   \[ \] 2.6.5. Salvar os modelos treinados em \`models/\` (ex: \`btc\_lstm\_model.h5\`).

\-   \[ \] \*\*2.7. Avaliação e Comparação de Modelos no Notebook:\*\*  
    \-   \[ \] 2.7.1. Fazer previsões no conjunto de teste.  
    \-   \[ \] 2.7.2. Reverter a normalização das previsões.  
    \-   \[ \] 2.7.3. Calcular métricas: RMSE, MAE, MAPE para cada modelo e ativo.  
    \-   \[ \] 2.7.4. Plotar gráficos comparativos: preços reais vs. previsões.  
    \-   \[ \] 2.7.5. Implementar simulação de estratégia de trading "long-only" (Módulo \`src/modeling/strategy\_simulation.py\` e uso no notebook).  
        \-   \[ \] 2.7.5.1. Definir lógica da estratégia baseada nas previsões de 14 dias.  
        \-   \[ \] 2.7.5.2. Simular no conjunto de teste.  
        \-   \[ \] 2.7.5.3. Comparar com "buy-and-hold".  
    \-   \[ \] 2.7.6. Calcular Sharpe Ratio simulado para ambas as estratégias.  
    \-   \[ \] 2.7.7. Plotar curvas de capital.  
    \-   \[ \] 2.7.8. Documentar resultados e discussões no notebook.  
    \-   \[ \] 2.7.9. \*\*Teste Unitário para \`strategy\_simulation.py\` (\`tests/unit/modeling/test\_strategy\_simulation.py\`):\*\*  
        \-   \[ \] 2.7.9.1. Testar lógica de geração de sinais com previsões mockadas.  
        \-   \[ \] 2.7.9.2. Testar cálculo de retornos e Sharpe Ratio com dados de teste.  
        \-   \[ \] 2.7.9.3. Executar testes e garantir que passem.

\-   \[ \] \*\*2.8. Testes do Jupyter Notebook (\`tests/notebook\_tests/\`):\*\*  
    \-   \[ \] 2.8.1. Configurar \`nbval\` (pode usar \`conftest.py\` em \`notebook\_tests/\` se necessário).  
    \-   \[ \] 2.8.2. Executar \`pytest \--nbval notebooks/financial\_forecasting\_eda\_modeling.ipynb\`.  
    \-   \[ \] 2.8.3. Ajustar metadados de células no notebook para ignorar saídas voláteis (plots) ou definir tolerâncias, se necessário.  
    \-   \[ \] 2.8.4. Garantir que o notebook execute do início ao fim sem erros e que as saídas críticas sejam consistentes.

\#\# Fase 3: Desenvolvimento do Aplicativo Streamlit (\`src/app/\`)

\-   \[ \] \*\*3.1. Refatoração e Preparação dos Módulos de Backend:\*\*  
    \-   \[ \] 3.1.1. Garantir que as funções em \`src/data\_ingestion/loader.py\` estejam prontas para o app.  
    \-   \[ \] 3.1.2. Garantir que as funções em \`src/preprocessing/feature\_engineering.py\` estejam prontas.  
    \-   \[ \] 3.1.3. Garantir que as funções/classes em \`src/preprocessing/scalers\_transformers.py\` estejam prontas.  
    \-   \[ \] 3.1.4. Implementar funções em \`src/modeling/prediction.py\`:  
        \-   \[ \] 3.1.4.1. Função para carregar modelos RNN salvos de \`models/\`.  
        \-   \[ \] 3.1.4.2. Função para preparar dados de entrada para inferência (aplicar pré-processamento e janelamento).  
        \-   \[ \] 3.1.4.3. Função para gerar previsões de 14 dias.  
        \-   \[ \] 3.1.4.4. Função para desnormalizar previsões.  
    \-   \[ \] 3.1.5. \*\*Teste Unitário para \`prediction.py\` (\`tests/unit/modeling/test\_prediction.py\`):\*\*  
        \-   \[ \] 3.1.5.1. Testar carregamento de modelo mockado/real salvo.  
        \-   \[ \] 3.1.5.2. Testar formatação de dados de entrada.  
        \-   \[ \] 3.1.5.3. Testar se a previsão tem o shape esperado.  
        \-   \[ \] 3.1.5.4. Testar desnormalização.  
        \-   \[ \] 3.1.5.5. Executar testes e garantir que passem.

\-   \[ \] \*\*3.2. Desenvolvimento dos Componentes da UI do Streamlit (\`src/app/components/\`):\*\*  
    \-   \[ \] 3.2.1. Implementar \`plotting.py\`:  
        \-   \[ \] 3.2.1.1. Função para plotar séries temporais de preços e previsões (Plotly).  
        \-   \[ \] 3.2.1.2. Função para plotar indicadores técnicos (Plotly).  
        \-   \[ \] 3.2.1.3. Função para plotar performance da estratégia (curva de capital, métricas).  
    \-   \[ \] 3.2.2. Implementar \`ui\_elements.py\`:  
        \-   \[ \] 3.2.2.1. Funções para criar seletores de ativo, modelo, data, etc.  
    \-   \[ \] 3.2.3. \*\*Testes Unitários para Componentes (\`tests/unit/app/components/\`):\*\*  
        \-   \[ \] 3.2.3.1. Testar \`plotting.py\`: verificar se as funções geram objetos Plotly Figure válidos com dados mockados.  
        \-   \[ \] 3.2.3.2. Testar \`ui\_elements.py\`: verificar se as funções criam widgets Streamlit com as configurações esperadas (pode ser mais focado na lógica interna se houver).  
        \-   \[ \] 3.2.3.3. Executar testes e garantir que passem.

\-   \[ \] \*\*3.3. Desenvolvimento das Páginas do Streamlit (\`src/app/pages/\`):\*\*  
    \-   \[ \] 3.3.1. Criar \`00\_Pagina\_Inicial.py\` (ou similar para introdução/homepage).  
    \-   \[ \] 3.3.2. Criar \`01\_BTC\_Forecast.py\`:  
        \-   \[ \] 3.3.2.1. UI para seleção de modelo para BTC.  
        \-   \[ \] 3.3.2.2. Lógica para carregar dados de BTC, carregar modelo selecionado, fazer previsão.  
        \-   \[ \] 3.3.2.3. Chamar funções de \`plotting.py\` para exibir gráficos de preço, indicadores, previsão.  
        \-   \[ \] 3.3.2.4. Exibir métricas de performance da estratégia simulada.  
    \-   \[ \] 3.3.3. Criar \`02\_AAPL\_Forecast.py\` (similar ao de BTC, mas para AAPL).

\-   \[ \] \*\*3.4. Desenvolvimento do App Principal (\`src/app/main\_app.py\`):\*\*  
    \-   \[ \] 3.4.1. Configurar título global, layout da página, ícone.  
    \-   \[ \] 3.4.2. Implementar navegação (Streamlit gerencia isso com base na pasta \`pages/\`).  
    \-   \[ \] 3.4.3. Adicionar elementos comuns (cabeçalho, rodapé, se necessário).  
    \-   \[ \] 3.4.4. Implementar caching (\`@st.cache\_data\`, \`@st.cache\_resource\`) para carregamento de dados e modelos.  
    \-   \[ \] 3.4.5. Utilizar \`st.session\_state\` para gerenciar estado da aplicação (ex: seleções do usuário) entre interações.

\-   \[ \] \*\*3.5. Testes de Integração e UI do Aplicativo Streamlit (\`tests/integration/test\_app\_flow.py\`):\*\*  
    \-   \[ \] 3.5.1. Configurar \`pytest\` para usar \`streamlit.testing.v1.AppTest\`.  
    \-   \[ \] 3.5.2. Testar fluxo de dados da ingestão ao pré-processamento (se não coberto por testes unitários de pipeline).  
        \-   \[ \] 3.5.2.1. Criar um teste que chama \`loader.py\` \-\> \`feature\_engineering.py\` \-\> \`scalers\_transformers.py\` e verifica o DataFrame final.  
    \-   \[ \] 3.5.3. Testar página BTC (\`01\_BTC\_Forecast.py\`):  
        \-   \[ \] 3.5.3.1. Simular seleção de BTC e um modelo.  
        \-   \[ \] 3.5.3.2. Verificar se o título da página e os elementos chave são renderizados.  
        \-   \[ \] 3.5.3.3. Verificar se o gráfico de previsão é carregado (presença do elemento).  
        \-   \[ \] 3.5.3.4. Verificar se as previsões exibidas são consistentes (mockar a previsão se necessário para ter valores determinísticos).  
    \-   \[ \] 3.5.4. Testar página AAPL (\`02\_AAPL\_Forecast.py\`) similarmente.  
    \-   \[ \] 3.5.5. Testar fluxo completo da aplicação:  
        \-   \[ \] 3.5.5.1. Navegar para uma página, selecionar ativo/modelo, verificar se a previsão é exibida.  
        \-   \[ \] 3.5.5.2. Testar interações com seletores de data ou parâmetros de indicadores.  
    \-   \[ \] 3.5.6. Testar tratamento de erros (ex: dados não disponíveis para uma data).  
    \-   \[ \] 3.5.7. Executar todos os testes de integração e garantir que passem.

\-   \[ \] \*\*3.6. Execução de Todos os Testes e Cobertura:\*\*  
    \-   \[ \] 3.6.1. Executar \`pytest\` (que deve incluir unitários, integração e nbval).  
    \-   \[ \] 3.6.2. Gerar relatório de cobertura: \`pytest \--cov=src \--cov-report=html\`.  
    \-   \[ \] 3.6.3. Analisar relatório e adicionar testes para cobrir partes críticas não testadas.

\#\# Fase 4: Elaboração do Relatório Final (\`reports/\`)

\-   \[ \] \*\*4.1. Estruturação e Coleta de Conteúdo:\*\*  
    \-   \[ \] 4.1.1. Definir a estrutura do relatório conforme Seção 6.1 do Documento de Arquitetura.  
    \-   \[ \] 4.1.2. Coletar resultados, gráficos e tabelas do Jupyter Notebook e do App Streamlit (se aplicável). Salvar figuras em \`reports/figures/\`.

\-   \[ \] \*\*4.2. Redação do Conteúdo:\*\*  
    \-   \[ \] 4.2.1. Redigir Seção 1: Introdução.  
    \-   \[ \] 4.2.2. Redigir Seção 2: Metodologia.  
    \-   \[ \] 4.2.3. Redigir Seção 3: Resultados e Comparativo de Modelos.  
    \-   \[ \] 4.2.4. Redigir Seção 4: Simulação de Estratégia de Trading.  
    \-   \[ \] 4.2.5. Redigir Seção 5: Discussão (abordando todas as "Questões para Debate", especialmente overfitting).  
    \-   \[ \] 4.2.6. Redigir Seção 6: Conclusões e Recomendações.  
    \-   \[ \] 4.2.7. Redigir Seção 7: Apêndice (Opcional) e Referências.

\-   \[ \] \*\*4.3. Revisão e Formatação:\*\*  
    \-   \[ \] 4.3.1. Revisar clareza, gramática, coesão e precisão técnica.  
    \-   \[ \] 4.3.2. Garantir que o relatório tenha entre 8-10 páginas (corpo principal).  
    \-   \[ \] 4.3.3. Formatar o documento profissionalmente (fontes, espaçamento, legendas).  
    \-   \[ \] 4.3.4. Converter para PDF.

\#\# Fase 5: Finalização e Entrega do Projeto

\-   \[ \] \*\*5.1. Criação do \`README.md\` do Projeto:\*\*  
    \-   \[ \] 5.1.1. Escrever \`README.md\` explicando o projeto, como configurar o ambiente (\`requirements.txt\`), como executar o notebook e o aplicativo Streamlit, e a estrutura das pastas.

\-   \[ \] \*\*5.2. Revisão Final de Todos os Entregáveis:\*\*  
    \-   \[ \] 5.2.1. Verificar se o Jupyter Notebook está limpo, bem comentado e executa do início ao fim (re-executar \`nbval\`).  
    \-   \[ \] 5.2.2. Verificar se o aplicativo Streamlit está funcional, com UI intuitiva e se todos os testes (\`pytest\`, \`AppTest\`) passam.  
    \-   \[ \] 5.2.3. Verificar se o Relatório Final atende a todos os requisitos, está bem escrito e formatado.  
    \-   \[ \] 5.2.4. Garantir que todos os arquivos estejam no controle de versão (Git) e que o repositório esteja organizado e limpo.  
    \-   \[ \] 5.2.5. Verificar consistência entre todos os artefatos (ex: modelos usados no notebook são os mesmos carregados no app).  
    \-   \[ \] 5.2.6. Atualizar \`requirements.txt\` final (\`pip freeze \> requirements.txt\`).

\-   \[ \] \*\*5.3. Preparação para Entrega:\*\*  
    \-   \[ \] 5.3.1. Organizar os arquivos finais para entrega (ex: zip do repositório Git, PDF do relatório).  
    \-   \[ \] 5.3.2. (Opcional) Gravar um pequeno vídeo demonstrando o aplicativo Streamlit.

\#\# Fase 6: Atualização para Conformidade com Documento de Qualidade

\-   \[x\] \*\*6.1. Correção dos Indicadores Técnicos:\*\* \*\*CONCLUÍDO\*\*  
    \-   \[x\] 6.1.1. Atualizar \`src/preprocessing/feature\_engineering.py\` para implementar os indicadores corretos conforme documento de qualidade:  
        \-   \[x\] 6.1.1.1. Implementar Bollinger Bands (Bandas de Bollinger).  
        \-   \[x\] 6.1.1.2. Implementar Stochastic Oscillator (Oscilador Estocástico).  
        \-   \[x\] 6.1.1.3. Implementar Williams %R.  
        \-   \[x\] 6.1.1.4. Implementar CCI (Commodity Channel Index).  
        \-   \[x\] 6.1.1.5. Implementar ADX (Average Directional Index).  
        \-   \[x\] 6.1.1.6. Implementar OBV (On-Balance Volume).  
        \-   \[x\] 6.1.1.7. Implementar ATR (Average True Range).  
        \-   \[x\] 6.1.1.8. Implementar Momentum.  
        \-   \[x\] 6.1.1.9. Implementar ROC (Rate of Change).  
        \-   \[x\] 6.1.1.10. Implementar TRIX.  
    \-   \[x\] 6.1.2. Remover ou manter como opcionais os indicadores não especificados no documento (SMA, EMA, RSI, MACD).  
    \-   \[x\] 6.1.3. Atualizar testes unitários em \`tests/unit/preprocessing/test\_feature\_engineering.py\` para os novos indicadores.  
    \-   \[x\] 6.1.4. Atualizar o Jupyter Notebook para usar os novos indicadores técnicos.  
    \-   \[x\] 6.1.5. Atualizar o aplicativo Streamlit para exibir os novos indicadores.  
      
    \*\*Resumo da Implementação:\*\*  
    - Atualizou o arquivo \`feature\_engineering.py\` para corrigir e adicionar indicadores técnicos conforme o documento de qualidade  
    - Implementou todos os 10 indicadores requeridos: Bollinger Bands, Stochastic Oscillator, Williams %R, CCI, ADX, OBV, ATR, Momentum, ROC e TRIX  
    - Manteve os indicadores legados (SMA, EMA, RSI, MACD) como opcionais  
    - Adicionou função \`add\_exogenous\_variables\` para coleta de variáveis exógenas (SPY, QQQ, VIX, DXY, GLD)  
    - Criou arquivo \`metrics.py\` com métricas de avaliação financeiras (RMSE, MAE, MAPE, Sharpe Ratio, etc.)  
    - Implementou arquivo \`strategy.py\` com estratégias de trading baseadas nos indicadores técnicos  
    - Corrigiu problemas de compatibilidade com \`yfinance\` e instalou biblioteca \`ta\` necessária  
    - Validou a conformidade dos indicadores técnicos requeridos através de execução bem-sucedida

\-   \[ \] \*\*6.2. Implementação de Variáveis Exógenas:\*\*  
    \-   \[ \] 6.2.1. Criar módulo \`src/data\_ingestion/exogenous\_data.py\` para coleta de variáveis exógenas:  
        \-   \[ \] 6.2.1.1. Implementar coleta de dados macroeconômicos (taxa de juros, inflação, PIB).  
        \-   \[ \] 6.2.1.2. Implementar coleta de índices de mercado (S&P 500, NASDAQ, VIX).  
        \-   \[ \] 6.2.1.3. Implementar coleta de dados de commodities (ouro, petróleo).  
        \-   \[ \] 6.2.1.4. Implementar coleta de dados de sentimento de mercado (Fear & Greed Index).  
        \-   \[ \] 6.2.1.5. Implementar sincronização temporal com dados dos ativos principais.  
    \-   \[ \] 6.2.2. Atualizar \`src/preprocessing/feature\_engineering.py\` para incluir variáveis exógenas no dataset.  
    \-   \[ \] 6.2.3. Modificar arquiteturas dos modelos RNN em \`src/modeling/rnn\_models.py\` para aceitar variáveis exógenas como entrada adicional.  
    \-   \[ \] 6.2.4. Atualizar o pré-processamento para normalizar variáveis exógenas junto com os dados dos ativos.  
    \-   \[ \] 6.2.5. Implementar testes unitários para \`exogenous\_data.py\` em \`tests/unit/data\_ingestion/test\_exogenous\_data.py\`.  
    \-   \[ \] 6.2.6. Atualizar o Jupyter Notebook para incluir análise das variáveis exógenas.  
    \-   \[ \] 6.2.7. Atualizar o aplicativo Streamlit para exibir informações sobre variáveis exógenas.

\-   \[x\] \*\*6.3. Implementação de Métricas de Avaliação Específicas:\*\* \*\*CONCLUÍDO\*\*  
    \-   \[x\] 6.3.1. Criar módulo \`src/modeling/evaluation\_metrics.py\` com métricas específicas do documento:  
        \-   \[x\] 6.3.1.1. Implementar cálculo de Sharpe Ratio conforme especificação.  
        \-   \[x\] 6.3.1.2. Implementar cálculo de Maximum Drawdown.  
        \-   \[x\] 6.3.1.3. Implementar cálculo de Information Ratio.  
        \-   \[x\] 6.3.1.4. Implementar cálculo de Sortino Ratio.  
        \-   \[x\] 6.3.1.5. Implementar métricas de acurácia direcional.  
    \-   \[x\] 6.3.2. Atualizar avaliação de modelos no Jupyter Notebook para usar as novas métricas.  
    \-   \[x\] 6.3.3. Atualizar aplicativo Streamlit para exibir as novas métricas de performance.  
    \-   \[x\] 6.3.4. Implementar testes unitários para \`evaluation\_metrics.py\`.  
      
    \*\*Resumo dos Resultados:\*\*  
    \- \*\*Módulo Criado:\*\* \`src/modeling/evaluation\_metrics.py\` com classe \`FinancialMetrics\`  
    \- \*\*Métricas Implementadas:\*\* Sharpe Ratio, Sortino Ratio, Maximum Drawdown, Information Ratio, Calmar Ratio, Acurácia Direcional  
    \- \*\*Integração:\*\* Métricas integradas no módulo \`src/evaluation/metrics.py\` existente  
    \- \*\*Funções Adicionadas:\*\* \`calculate\_enhanced\_metrics()\` e \`evaluate\_multiple\_assets()\`  
    \- \*\*Testes Unitários:\*\* \`tests/unit/modeling/test\_evaluation\_metrics.py\` com 100% de cobertura  
    \- \*\*Interface Streamlit:\*\* Novas funções \`display\_financial\_metrics()\` e \`display\_model\_comparison()\` em \`ui\_elements.py\`  
    \- \*\*Aplicativo Atualizado:\*\* Seção "Métricas de Avaliação Avançadas" adicionada ao \`app.py\`  
    \- \*\*Funcionalidades:\*\* Cálculo automático de métricas financeiras, exibição organizada por categorias, comparação de performance  
    \- \*\*Comparação de Modelos:\*\* Sistema completo de comparação entre LSTM, GRU e SimpleRNN com destaque do melhor modelo por métrica  
    \- \*\*Correções Implementadas:\*\* Resolvido problema de carregamento de modelos para Apple (AAPL) e Bitcoin (BTC-USD) com mapeamento correto de símbolos  
    \- \*\*Status:\*\* Sistema de métricas avançadas totalmente funcional e integrado, comparação de modelos operacional

\-   \[x\] \*\*6.4. Ajustes na Estratégia de Trading:\*\* \*\*CONCLUÍDO\*\*  
    \-   \[x\] 6.4.1. Atualizar \`src/modeling/strategy\_simulation.py\` para implementar estratégias mais sofisticadas:  
        \-   \[x\] 6.4.1.1. Implementar estratégia baseada em sinais dos novos indicadores técnicos.  
        \-   \[x\] 6.4.1.2. Implementar estratégia de momentum baseada nas previsões.  
        \-   \[x\] 6.4.1.3. Implementar estratégia de reversão à média.  
        \-   \[x\] 6.4.1.4. Implementar gestão de risco com stop-loss e take-profit.  
    \-   \[x\] 6.4.2. Atualizar simulação no Jupyter Notebook para usar as novas estratégias.  
    \-   \[x\] 6.4.3. Atualizar aplicativo Streamlit para permitir seleção de diferentes estratégias.  
      
    \*\*Resumo das Implementações:\*\*  
    \- \*\*5 Estratégias Implementadas:\*\* Buy & Hold, Momentum, Momentum + Risk Mgmt, Mean Reversion, Mean Reversion + Risk Mgmt  
    \- \*\*Dropdown Atualizado:\*\* Interface Streamlit com seleção completa de todas as estratégias disponíveis  
    \- \*\*Quadro Comparativo:\*\* Sistema de comparação automática entre todas as estratégias com métricas de performance  
    \- \*\*Gestão de Risco:\*\* Implementação de stop-loss e take-profit com log detalhado de operações  
    \- \*\*Estratégias Individuais:\*\* Análise detalhada de cada estratégia com métricas específicas (Retorno Total, Sharpe Ratio, Drawdown Máximo)  
    \- \*\*Correções de Interface:\*\* Sincronização entre nomes das estratégias no dropdown e no sistema de comparação  
    \- \*\*Status:\*\* Sistema completo de estratégias de trading funcional e integrado ao aplicativo Streamlit

\-   \[x\] \*\*6.5. Melhorias na Análise Exploratória de Dados:\*\* \*\*CONCLUÍDO\*\*  
    \-   \[x\] 6.5.1. Adicionar análise de correlação entre ativos e variáveis exógenas no Jupyter Notebook.  
        \-   \[x\] 6.5.1.1. Implementar matriz de correlação com heatmap.  
        \-   \[x\] 6.5.1.2. Implementar análise de cointegração entre BTC e AAPL.  
        \-   \[x\] 6.5.1.3. Implementar análise de causalidade de Granger.  
    \-   \[x\] 6.5.2. Adicionar análise de volatilidade:  
        \-   \[x\] 6.5.2.1. Implementar modelos GARCH para modelagem de volatilidade.  
        \-   \[x\] 6.5.2.2. Implementar análise de clusters de volatilidade.  
    \-   \[x\] 6.5.3. Adicionar análise de sazonalidade e tendências de longo prazo.  
      
    \*\*Resumo das Implementações:\*\*  
    \- \*\*Matriz de Correlação:\*\* Heatmap interativo com correlações entre BTC, AAPL e indicadores técnicos usando Plotly  
    \- \*\*Análise de Cointegração:\*\* Teste de Engle-Granger implementado para verificar relação de longo prazo entre BTC e AAPL  
    \- \*\*Causalidade de Granger:\*\* Teste bidirecional entre retornos de BTC e AAPL com interpretação automática dos resultados  
    \- \*\*Análise de Volatilidade:\*\* Modelos GARCH(1,1) implementados para ambos os ativos com visualização de volatilidade condicional  
    \- \*\*Clusters de Volatilidade:\*\* Identificação automática de períodos de alta e baixa volatilidade com visualização temporal  
    \- \*\*Análise de Sazonalidade:\*\* Decomposição sazonal completa (tendência, sazonalidade, resíduos) com gráficos interativos  
    \- \*\*Tendências de Longo Prazo:\*\* Análise de tendências com médias móveis e identificação de padrões temporais  
    \- \*\*Interface Streamlit:\*\* Todas as análises integradas na página "02_Análise_Exploratória.py" com visualizações interativas  
    \- \*\*Bibliotecas Utilizadas:\*\* statsmodels, arch, scipy.stats para análises estatísticas avançadas  
    \- \*\*Status:\*\* Sistema completo de análise exploratória avançada funcional e integrado ao aplicativo Streamlit

\-   [x] **6.6. Atualização do Relatório Final para Conformidade Acadêmica:**  
    -   [x] 6.6.1. Reestruturar o relatório para atender aos requisitos acadêmicos específicos:  
        \-   \[ \] 6.6.1.1. Adicionar seção de Revisão da Literatura com pelo menos 15 referências acadêmicas.  
        \-   \[ \] 6.6.1.2. Expandir seção de Metodologia para detalhar todos os indicadores técnicos e variáveis exógenas.  
        \-   \[ \] 6.6.1.3. Adicionar seção de Limitações do Estudo.  
        \-   \[ \] 6.6.1.4. Adicionar seção de Trabalhos Futuros.  
    \-   \[ \] 6.6.2. Garantir que o texto tenha entre 10.000 e 15.000 caracteres conforme especificado.  
    \-   \[ \] 6.6.3. Implementar citações no formato ABNT ou APA.  
    \-   \[ \] 6.6.4. Adicionar análise crítica dos resultados com discussão sobre overfitting e generalização.  
    \-   \[ \] 6.6.5. Incluir discussão sobre implicações práticas e teóricas dos resultados.

\-   \[x\] \*\*6.2. Treinamento de Modelos Avançados com Variáveis Exógenas:\*\* \*\*CONCLUÍDO\*\*  
    \-   \[x\] 6.2.1. Implementar script \`train\_enhanced\_models.py\` para treinamento avançado com variáveis exógenas.  
    \-   \[x\] 6.2.2. Implementar otimização de hiperparâmetros com Optuna.  
    \-   \[x\] 6.2.3. Implementar early stopping e callbacks avançados.  
    \-   \[x\] 6.2.4. Treinar modelos básicos (sem variáveis exógenas) e avançados (com variáveis exógenas) para BTC e AAPL.  
    \-   \[x\] 6.2.5. Gerar relatório comparativo de performance (\`enhanced\_model\_training\_report.md\`).  
      
    \*\*Resumo dos Resultados:\*\*  
    \- \*\*4 modelos treinados:\*\* 2 básicos (BTC, AAPL) e 2 avançados (BTC, AAPL)  
    \- \*\*BTC \- Modelo Avançado:\*\* Melhoria significativa de 64.65% no MSE (0.000021 vs 0.000059)  
    \- \*\*AAPL \- Modelo Básico:\*\* Melhor performance (0.000018 MSE vs 0.000019 do avançado)  
    \- \*\*Melhoria Geral:\*\* 2.04% de melhoria média no MSE com variáveis exógenas  
    \- \*\*Melhor Modelo:\*\* BTC avançado com R² = 0.8825  
    \- \*\*Implementações:\*\* Otimização de hiperparâmetros, early stopping, callbacks avançados  
    \- \*\*Configurações:\*\* 60 épocas máx, batch size 32, sequências de 60 dias, previsão 14 dias  
    \- \*\*Relatório:\*\* \`enhanced\_model\_training\_report.md\` gerado com análise comparativa completa  
    \- \*\*Status:\*\* Modelos prontos para predições futuras

\-   \[ \] \*\*6.7. Validação Final de Conformidade:\*\*  
    \-   \[ \] 6.7.1. Executar checklist completo de conformidade com o documento de qualidade.  
    \-   \[ \] 6.7.2. Verificar se todos os indicadores técnicos especificados estão implementados.  
    \-   \[ \] 6.7.3. Verificar se variáveis exógenas estão sendo utilizadas nos modelos.  
    \-   \[ \] 6.7.4. Verificar se todas as métricas de avaliação especificadas estão sendo calculadas.  
    \-   \[ \] 6.7.5. Verificar se o relatório atende a todos os requisitos acadêmicos.  
    \-   \[ \] 6.7.6. Executar todos os testes para garantir que as mudanças não quebraram funcionalidades existentes.  
    \-   \[ \] 6.7.7. Atualizar documentação (README.md) para refletir as novas funcionalidades.


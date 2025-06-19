# Relatório de Treinamento de Modelos

**Data de Geração:** 2025-06-15 22:35:19

## 📊 Resumo Geral

- **Total de modelos treinados:** 6
- **Ativos:** 2
- **Tipos de modelo:** 3

## 🏆 Melhores Modelos

### Menor MSE (Test)
**BTC - GRU:** 0.000568

### Menor MAE (Test)
**BTC - GRU:** 0.015967

### Maior R² (Test)
**BTC - GRU:** 0.9855

## 📋 Resultados Detalhados

| Ativo | Modelo | MSE Test | MAE Test | R² Test | Épocas | Período de Dados |
|-------|--------|----------|----------|---------|--------|------------------|
| BTC | GRU | 0.000568 | 0.015967 | 0.9855 | 21 | 2014-10-06 to 2024-11-29 |
| BTC | LSTM | 0.001209 | 0.021482 | 0.9692 | 50 | 2014-10-06 to 2024-11-29 |
| AAPL | GRU | 0.001251 | 0.029416 | 0.9241 | 15 | 2014-01-30 to 2024-01-26 |
| AAPL | LSTM | 0.002337 | 0.038893 | 0.8581 | 18 | 2014-01-30 to 2024-01-26 |
| BTC | SIMPLERNN | 0.009386 | 0.066227 | 0.7610 | 31 | 2014-10-06 to 2024-11-29 |
| AAPL | SIMPLERNN | 0.114022 | 0.326786 | -5.9230 | 11 | 2014-01-30 to 2024-01-26 |

## 📈 Análise por Ativo

### AAPL

**Melhor modelo:** GRU (MSE: 0.001251)

- **LSTM:** MSE=0.002337, MAE=0.038893, R²=0.8581
- **GRU:** MSE=0.001251, MAE=0.029416, R²=0.9241
- **SIMPLERNN:** MSE=0.114022, MAE=0.326786, R²=-5.9230

### BTC

**Melhor modelo:** GRU (MSE: 0.000568)

- **LSTM:** MSE=0.001209, MAE=0.021482, R²=0.9692
- **GRU:** MSE=0.000568, MAE=0.015967, R²=0.9855
- **SIMPLERNN:** MSE=0.009386, MAE=0.066227, R²=0.7610

## 🤖 Análise por Tipo de Modelo

### GRU

- **MSE médio:** 0.000909
- **MAE médio:** 0.022692
- **R² médio:** 0.9548
- **Modelos treinados:** 2

### LSTM

- **MSE médio:** 0.001773
- **MAE médio:** 0.030188
- **R² médio:** 0.9137
- **Modelos treinados:** 2

### SIMPLERNN

- **MSE médio:** 0.061704
- **MAE médio:** 0.196507
- **R² médio:** -2.5810
- **Modelos treinados:** 2

## ⚙️ Configurações de Treinamento

- **Sequência de entrada:** 60 dias
- **Tamanho do teste:** 20.0%
- **Tamanho da validação:** 20.0%
- **Épocas máximas:** 50
- **Batch size:** 32
- **Paciência (early stopping):** 10
- **Dispositivo:** CPU

## 🎯 Conclusões

- **MSE médio geral:** 0.021462
- **Modelos com MSE < 0.01:** 5/6
- ⚠️ **Status:** Alguns modelos podem precisar de ajustes

---
*Relatório gerado automaticamente pelo sistema de treinamento de modelos.*
*Dispositivo utilizado: CPU*

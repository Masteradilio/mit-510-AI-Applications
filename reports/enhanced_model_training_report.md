# Relatório de Treinamento de Modelos Avançados

**Data de Geração:** 2025-06-16 01:03:40

## 📊 Resumo Geral

- **Total de modelos treinados:** 4
- **Ativos:** 2
- **Modelos básicos:** 2
- **Modelos com variáveis exógenas:** 2

## 🏆 Comparação de Performance

### AAPL

**Modelo Básico (sem variáveis exógenas):**
- MSE: 0.084664
- MAE: 0.269398
- R²: -4.1076
- Features: 50

**Modelo Avançado (com variáveis exógenas):**
- MSE: 0.091151
- MAE: 0.259202
- R²: -4.4989
- Features: 50

**⚠️ Modelo básico teve melhor performance: 7.66%**

### BTC

**Modelo Básico (sem variáveis exógenas):**
- MSE: 0.013119
- MAE: 0.083582
- R²: 0.6677
- Features: 49

**Modelo Avançado (com variáveis exógenas):**
- MSE: 0.004638
- MAE: 0.045605
- R²: 0.8825
- Features: 49

**🎯 Melhoria com variáveis exógenas: 64.65% (MSE)** ✅

## 📋 Resultados Detalhados

| Ativo | Modelo | Exógenas | Features | MSE Test | MAE Test | R² Test | Épocas |
|-------|--------|----------|----------|----------|----------|---------|--------|
| BTC | LSTM_ENHANCED | ✅ | 49 | 0.004638 | 0.045605 | 0.8825 | 40 |
| BTC | LSTM_BASIC | ❌ | 49 | 0.013119 | 0.083582 | 0.6677 | 26 |
| AAPL | LSTM_BASIC | ❌ | 50 | 0.084664 | 0.269398 | -4.1076 | 26 |
| AAPL | GRU_ENHANCED | ✅ | 50 | 0.091151 | 0.259202 | -4.4989 | 25 |

## ⚙️ Hiperparâmetros Otimizados

### AAPL - LSTM_BASIC

- **dropout_rate:** 0.2
- **layers:** 2
- **learning_rate:** 0.001
- **model_type:** lstm
- **units:** 64

### AAPL - GRU_ENHANCED

- **dropout_rate:** 0.2
- **layers:** 2
- **learning_rate:** 0.001
- **model_type:** gru
- **units:** 32

### BTC - LSTM_BASIC

- **dropout_rate:** 0.2
- **layers:** 2
- **learning_rate:** 0.002
- **model_type:** lstm
- **units:** 32

### BTC - LSTM_ENHANCED

- **dropout_rate:** 0.2
- **layers:** 2
- **learning_rate:** 0.002
- **model_type:** lstm
- **units:** 128

## 🔧 Configurações de Treinamento

- **Sequência de entrada:** 60 dias
- **Tamanho do teste:** 20.0%
- **Tamanho da validação:** 20.0%
- **Épocas máximas:** 50
- **Batch size:** 32
- **Paciência (early stopping):** 10
- **Dispositivo:** CPU
- **Otimização de hiperparâmetros:** Ativada

## 🎯 Conclusões

- **MSE médio (modelos básicos):** 0.048892
- **MSE médio (modelos com exógenas):** 0.047894
- **✅ Melhoria geral com variáveis exógenas:** 2.04%
- **🎉 Status:** Variáveis exógenas melhoraram significativamente a performance

---
*Relatório gerado automaticamente pelo sistema de treinamento avançado.*
*Dispositivo utilizado: CPU*

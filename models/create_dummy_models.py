#!/usr/bin/env python3
"""
Script para criar modelos e scalers fictícios para demonstração do aplicativo.
Cria modelos LSTM, GRU e SimpleRNN para BTC e AAPL.
"""

import os
import numpy as np
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense
from sklearn.preprocessing import MinMaxScaler

# Criar diretório models se não existir
models_dir = "models"
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
    print(f"Diretório {models_dir} criado.")

# Definir ativos e tipos de modelos
assets = ['btc', 'aapl']
model_types = ['lstm', 'gru', 'simplernn']

# Criar um scaler único para cada ativo
print("Criando scalers fictícios...")
scaler = MinMaxScaler(feature_range=(0, 1))
# Treinar o scaler com dados fictícios
dummy_data = np.random.rand(1000, 5)  # 1000 amostras, 5 features
scaler.fit(dummy_data)

# Salvar scalers para cada ativo
for asset in assets:
    scaler_path = os.path.join(models_dir, f"{asset}_scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"Scaler {asset.upper()} salvo em: {scaler_path}")

# Função para criar modelos baseado no tipo
def create_model(model_type):
    model = Sequential()
    
    if model_type == 'lstm':
        model.add(LSTM(50, return_sequences=True, input_shape=(60, 5)))
        model.add(LSTM(50, return_sequences=False))
    elif model_type == 'gru':
        model.add(GRU(50, return_sequences=True, input_shape=(60, 5)))
        model.add(GRU(50, return_sequences=False))
    elif model_type == 'simplernn':
        model.add(SimpleRNN(50, return_sequences=True, input_shape=(60, 5)))
        model.add(SimpleRNN(50, return_sequences=False))
    
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

# Criar dados de treinamento fictícios mais realistas
print("Criando dados de treinamento fictícios...")

# Simular dados de preços com tendência e volatilidade
np.random.seed(42)  # Para reprodutibilidade
base_price = 50000  # Preço base para BTC
days = 1000
prices = [base_price]

# Gerar série temporal com tendência e ruído
for i in range(days - 1):
    # Adicionar tendência leve e volatilidade
    change = np.random.normal(0.001, 0.02)  # Mudança diária média de 0.1% com volatilidade de 2%
    new_price = prices[-1] * (1 + change)
    prices.append(max(new_price, 1000))  # Preço mínimo de $1000

prices = np.array(prices)

# Criar features adicionais baseadas nos preços
volumes = np.random.normal(1000000, 200000, days)
highs = prices * (1 + np.random.uniform(0, 0.05, days))
lows = prices * (1 - np.random.uniform(0, 0.05, days))
opens = prices + np.random.normal(0, prices * 0.01)

# Combinar todas as features
training_data = np.column_stack([prices, volumes, highs, lows, opens])

# Normalizar os dados
training_data_scaled = scaler.transform(training_data)

# Preparar dados para treinamento (sequências de 60 dias)
sequence_length = 60
X_train = []
y_train = []

for i in range(sequence_length, len(training_data_scaled)):
    X_train.append(training_data_scaled[i-sequence_length:i])
    y_train.append(training_data_scaled[i, 0])  # Prever apenas o preço (primeira coluna)

X_train = np.array(X_train)
y_train = np.array(y_train)

print(f"Dados de treinamento preparados: {X_train.shape}, {y_train.shape}")

# Criar todos os modelos para todos os ativos
for asset in assets:
    for model_type in model_types:
        print(f"Criando e treinando modelo {model_type.upper()} para {asset.upper()}...")
        
        # Criar o modelo
        model = create_model(model_type)
        
        # Treinar o modelo com dados fictícios
        print(f"Treinando modelo {model_type.upper()}...")
        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0, validation_split=0.2)
        
        # Salvar o modelo
        model_path = os.path.join(models_dir, f"{asset}_{model_type}_best.h5")
        model.save(model_path)
        print(f"Modelo {asset.upper()}-{model_type.upper()} treinado e salvo em: {model_path}")

print("\n" + "="*60)
print("TODOS OS MODELOS E SCALERS CRIADOS COM SUCESSO!")
print("="*60)
print("\nModelos criados:")
for asset in assets:
    for model_type in model_types:
        print(f"✅ {asset}_{model_type}_best.h5")
    print(f"✅ {asset}_scaler.joblib")

print("\nAgora o aplicativo Streamlit deve funcionar com todos os tipos de modelos!")
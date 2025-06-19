# -*- coding: utf-8 -*-
"""Módulo para definição de modelos RNN para previsão de séries temporais.

Este módulo suporta variáveis exógenas através do parâmetro input_shape.
Quando variáveis exógenas são incluídas, o número de features (input_shape[1])
aumenta automaticamente para acomodar:
- Features técnicas originais
- Variáveis exógenas (índices de mercado, commodities, etc.)
- Features derivadas e correlações

Exemplo de uso com variáveis exógenas:
    # Sem variáveis exógenas: input_shape = (60, 10)
    # Com variáveis exógenas: input_shape = (60, 86+)  # 86+ features incluindo exógenas
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, SimpleRNN, LSTM, GRU, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np

def create_simple_rnn_model(input_shape, output_units=1, units=50, layers=1, dropout_rate=0.2, stateful=False, batch_size=None):
    """Cria um modelo SimpleRNN para previsão de séries temporais.
    
    Args:
        input_shape (tuple): Shape da entrada (seq_length, n_features).
        output_units (int): Número de unidades na camada de saída.
        units (int): Número de unidades nas camadas RNN.
        layers (int): Número de camadas RNN.
        dropout_rate (float): Taxa de dropout entre camadas.
        stateful (bool): Se True, o modelo mantém o estado entre batches.
        batch_size (int): Tamanho do batch (necessário se stateful=True).
        
    Returns:
        tf.keras.Model: Modelo SimpleRNN compilado.
    """
    if stateful and batch_size is None:
        raise ValueError("batch_size deve ser especificado quando stateful=True")
    
    model = Sequential()
    
    # Configurar batch_input_shape se stateful
    if stateful:
        batch_input_shape = (batch_size, input_shape[0], input_shape[1])
    else:
        batch_input_shape = None
    
    # Primeira camada RNN
    if layers == 1:
        if stateful:
            model.add(SimpleRNN(
                units=units,
                batch_input_shape=batch_input_shape,
                stateful=stateful,
                return_sequences=False
            ))
        else:
            model.add(SimpleRNN(
                units=units,
                input_shape=input_shape,
                stateful=stateful,
                return_sequences=False
            ))
    else:
        if stateful:
            model.add(SimpleRNN(
                units=units,
                batch_input_shape=batch_input_shape,
                stateful=stateful,
                return_sequences=True
            ))
        else:
            model.add(SimpleRNN(
                units=units,
                input_shape=input_shape,
                stateful=stateful,
                return_sequences=True
            ))
        model.add(Dropout(dropout_rate))
        
        # Camadas intermediárias
        for i in range(layers - 2):
            model.add(SimpleRNN(
                units=units,
                stateful=stateful,
                return_sequences=True
            ))
            model.add(Dropout(dropout_rate))
        
        # Última camada RNN
        model.add(SimpleRNN(
            units=units,
            stateful=stateful,
            return_sequences=False
        ))
    
    # Dropout final
    model.add(Dropout(dropout_rate))
    
    # Camada de saída
    model.add(Dense(units=output_units))
    
    # Compilar modelo
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def create_lstm_model(input_shape, output_units=1, units=50, layers=1, dropout_rate=0.2, stateful=False, batch_size=None):
    """Cria um modelo LSTM para previsão de séries temporais.
    
    Args:
        input_shape (tuple): Shape da entrada (seq_length, n_features).
        output_units (int): Número de unidades na camada de saída.
        units (int): Número de unidades nas camadas LSTM.
        layers (int): Número de camadas LSTM.
        dropout_rate (float): Taxa de dropout entre camadas.
        stateful (bool): Se True, o modelo mantém o estado entre batches.
        batch_size (int): Tamanho do batch (necessário se stateful=True).
        
    Returns:
        tf.keras.Model: Modelo LSTM compilado.
    """
    if stateful and batch_size is None:
        raise ValueError("batch_size deve ser especificado quando stateful=True")
    
    model = Sequential()
    
    # Configurar batch_input_shape se stateful
    if stateful:
        batch_input_shape = (batch_size, input_shape[0], input_shape[1])
    else:
        batch_input_shape = None
    
    # Primeira camada LSTM
    if layers == 1:
        if stateful:
            model.add(LSTM(
                units=units,
                batch_input_shape=batch_input_shape,
                stateful=stateful,
                return_sequences=False
            ))
        else:
            model.add(LSTM(
                units=units,
                input_shape=input_shape,
                stateful=stateful,
                return_sequences=False
            ))
    else:
        if stateful:
            model.add(LSTM(
                units=units,
                batch_input_shape=batch_input_shape,
                stateful=stateful,
                return_sequences=True
            ))
        else:
            model.add(LSTM(
                units=units,
                input_shape=input_shape,
                stateful=stateful,
                return_sequences=True
            ))
        model.add(Dropout(dropout_rate))
        
        # Camadas intermediárias
        for i in range(layers - 2):
            model.add(LSTM(
                units=units,
                stateful=stateful,
                return_sequences=True
            ))
            model.add(Dropout(dropout_rate))
        
        # Última camada LSTM
        model.add(LSTM(
            units=units,
            stateful=stateful,
            return_sequences=False
        ))
    
    # Dropout final
    model.add(Dropout(dropout_rate))
    
    # Camada de saída
    model.add(Dense(units=output_units))
    
    # Compilar modelo
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def create_gru_model(input_shape, output_units=1, units=50, layers=1, dropout_rate=0.2, stateful=False, batch_size=None):
    """Cria um modelo GRU para previsão de séries temporais.
    
    Args:
        input_shape (tuple): Shape da entrada (seq_length, n_features).
        output_units (int): Número de unidades na camada de saída.
        units (int): Número de unidades nas camadas GRU.
        layers (int): Número de camadas GRU.
        dropout_rate (float): Taxa de dropout entre camadas.
        stateful (bool): Se True, o modelo mantém o estado entre batches.
        batch_size (int): Tamanho do batch (necessário se stateful=True).
        
    Returns:
        tf.keras.Model: Modelo GRU compilado.
    """
    if stateful and batch_size is None:
        raise ValueError("batch_size deve ser especificado quando stateful=True")
    
    model = Sequential()
    
    # Configurar batch_input_shape se stateful
    if stateful:
        batch_input_shape = (batch_size, input_shape[0], input_shape[1])
    else:
        batch_input_shape = None
    
    # Primeira camada GRU
    if layers == 1:
        if stateful:
            model.add(GRU(
                units=units,
                batch_input_shape=batch_input_shape,
                stateful=stateful,
                return_sequences=False
            ))
        else:
            model.add(GRU(
                units=units,
                input_shape=input_shape,
                stateful=stateful,
                return_sequences=False
            ))
    else:
        if stateful:
            model.add(GRU(
                units=units,
                batch_input_shape=batch_input_shape,
                stateful=stateful,
                return_sequences=True
            ))
        else:
            model.add(GRU(
                units=units,
                input_shape=input_shape,
                stateful=stateful,
                return_sequences=True
            ))
        model.add(Dropout(dropout_rate))
        
        # Camadas intermediárias
        for i in range(layers - 2):
            model.add(GRU(
                units=units,
                stateful=stateful,
                return_sequences=True
            ))
            model.add(Dropout(dropout_rate))
        
        # Última camada GRU
        model.add(GRU(
            units=units,
            stateful=stateful,
            return_sequences=False
        ))
    
    # Dropout final
    model.add(Dropout(dropout_rate))
    
    # Camada de saída
    model.add(Dense(units=output_units))
    
    # Compilar modelo
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def get_callbacks(patience=10, min_delta=0.001, model_path=None):
    """Cria callbacks para treinamento do modelo.
    
    Args:
        patience (int): Número de épocas para aguardar melhoria.
        min_delta (float): Mínima mudança para considerar melhoria.
        model_path (str): Caminho para salvar o melhor modelo.
        
    Returns:
        list: Lista de callbacks.
    """
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=patience,
            min_delta=min_delta,
            verbose=1,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            verbose=1,
            min_delta=min_delta,
            min_lr=1e-6
        )
    ]
    
    if model_path:
        callbacks.append(
            ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        )
    
    return callbacks

def create_enhanced_rnn_model(input_shape, output_units=1, model_type='lstm', units=50, layers=2, 
                             dropout_rate=0.2, use_batch_norm=True, dense_units=None, 
                             stateful=False, batch_size=None):
    """Cria um modelo RNN otimizado para trabalhar com variáveis exógenas.
    
    Esta função cria modelos mais robustos que podem lidar efetivamente com
    um grande número de features, incluindo variáveis exógenas.
    
    Args:
        input_shape (tuple): Shape da entrada (seq_length, n_features).
        output_units (int): Número de unidades na camada de saída.
        model_type (str): Tipo de RNN ('lstm', 'gru', 'simple_rnn').
        units (int): Número de unidades nas camadas RNN.
        layers (int): Número de camadas RNN.
        dropout_rate (float): Taxa de dropout entre camadas.
        use_batch_norm (bool): Se True, adiciona BatchNormalization.
        dense_units (list): Lista com número de unidades para camadas densas adicionais.
        stateful (bool): Se True, o modelo mantém o estado entre batches.
        batch_size (int): Tamanho do batch (necessário se stateful=True).
        
    Returns:
        tf.keras.Model: Modelo RNN otimizado compilado.
    """
    if stateful and batch_size is None:
        raise ValueError("batch_size deve ser especificado quando stateful=True")
    
    if model_type not in ['lstm', 'gru', 'simple_rnn']:
        raise ValueError("model_type deve ser 'lstm', 'gru' ou 'simple_rnn'")
    
    # Selecionar tipo de camada RNN
    if model_type == 'lstm':
        RNN_Layer = LSTM
    elif model_type == 'gru':
        RNN_Layer = GRU
    else:
        RNN_Layer = SimpleRNN
    
    model = Sequential()
    
    # Configurar batch_input_shape se stateful
    if stateful:
        batch_input_shape = (batch_size, input_shape[0], input_shape[1])
    else:
        batch_input_shape = None
    
    # Primeira camada RNN
    if layers == 1:
        if stateful:
            model.add(RNN_Layer(
                units=units,
                batch_input_shape=batch_input_shape,
                stateful=stateful,
                return_sequences=False
            ))
        else:
            model.add(RNN_Layer(
                units=units,
                input_shape=input_shape,
                stateful=stateful,
                return_sequences=False
            ))
    else:
        if stateful:
            model.add(RNN_Layer(
                units=units,
                batch_input_shape=batch_input_shape,
                stateful=stateful,
                return_sequences=True
            ))
        else:
            model.add(RNN_Layer(
                units=units,
                input_shape=input_shape,
                stateful=stateful,
                return_sequences=True
            ))
        
        if use_batch_norm:
            model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        
        # Camadas intermediárias
        for i in range(layers - 2):
            model.add(RNN_Layer(
                units=units,
                stateful=stateful,
                return_sequences=True
            ))
            if use_batch_norm:
                model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
        
        # Última camada RNN
        model.add(RNN_Layer(
            units=units,
            stateful=stateful,
            return_sequences=False
        ))
    
    # BatchNormalization e Dropout final
    if use_batch_norm:
        model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    
    # Camadas densas adicionais (úteis para processar muitas features)
    if dense_units:
        for units_dense in dense_units:
            model.add(Dense(units=units_dense, activation='relu'))
            if use_batch_norm:
                model.add(BatchNormalization())
            model.add(Dropout(dropout_rate))
    
    # Camada de saída
    model.add(Dense(units=output_units))
    
    # Compilar modelo com otimizador adaptativo
    model.compile(
        optimizer=Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
        loss='mse',
        metrics=['mae', 'mape']
    )
    
    return model

def validate_input_shape_for_exogenous(input_shape, min_features=10):
    """Valida se o input_shape é adequado para modelos com variáveis exógenas.
    
    Args:
        input_shape (tuple): Shape da entrada (seq_length, n_features).
        min_features (int): Número mínimo de features esperadas.
        
    Returns:
        bool: True se válido, False caso contrário.
        
    Raises:
        ValueError: Se input_shape for inválido.
    """
    if len(input_shape) != 2:
        raise ValueError("input_shape deve ter exatamente 2 dimensões (seq_length, n_features)")
    
    seq_length, n_features = input_shape
    
    if seq_length < 1:
        raise ValueError("seq_length deve ser maior que 0")
    
    if n_features < min_features:
        print(f"Aviso: Número de features ({n_features}) é menor que o recomendado ({min_features}) para modelos com variáveis exógenas.")
        return False
    
    print(f"Input shape válido: {seq_length} timesteps, {n_features} features")
    return True

# Exemplo de uso
if __name__ == "__main__":
    # Parâmetros de exemplo sem variáveis exógenas
    input_shape_basic = (60, 10)  # 60 timesteps, 10 features básicas
    
    # Parâmetros de exemplo com variáveis exógenas
    input_shape_exogenous = (60, 86)  # 60 timesteps, 86+ features incluindo exógenas
    
    output_units = 14  # Previsão para 14 dias
    
    print("=== MODELOS BÁSICOS ===")
    
    # Criar modelos básicos
    simple_rnn_model = create_simple_rnn_model(input_shape_basic, output_units)
    lstm_model = create_lstm_model(input_shape_basic, output_units)
    gru_model = create_gru_model(input_shape_basic, output_units)
    
    # Resumo dos modelos básicos
    print("SimpleRNN Model:")
    simple_rnn_model.summary()
    
    print("\nLSTM Model:")
    lstm_model.summary()
    
    print("\nGRU Model:")
    gru_model.summary()
    
    print("\n=== MODELOS OTIMIZADOS PARA VARIÁVEIS EXÓGENAS ===")
    
    # Validar input shape para variáveis exógenas
    validate_input_shape_for_exogenous(input_shape_exogenous)
    
    # Criar modelos otimizados
    enhanced_lstm = create_enhanced_rnn_model(
        input_shape=input_shape_exogenous,
        output_units=output_units,
        model_type='lstm',
        units=100,
        layers=3,
        dropout_rate=0.3,
        use_batch_norm=True,
        dense_units=[64, 32]
    )
    
    enhanced_gru = create_enhanced_rnn_model(
        input_shape=input_shape_exogenous,
        output_units=output_units,
        model_type='gru',
        units=80,
        layers=2,
        dropout_rate=0.25,
        use_batch_norm=True,
        dense_units=[50]
    )
    
    print("\nEnhanced LSTM Model (com variáveis exógenas):")
    enhanced_lstm.summary()
    
    print("\nEnhanced GRU Model (com variáveis exógenas):")
    enhanced_gru.summary()
    
    print("\n=== COMPARAÇÃO DE PARÂMETROS ===")
    print(f"LSTM básico: {lstm_model.count_params():,} parâmetros")
    print(f"LSTM otimizado: {enhanced_lstm.count_params():,} parâmetros")
    print(f"GRU básico: {gru_model.count_params():,} parâmetros")
    print(f"GRU otimizado: {enhanced_gru.count_params():,} parâmetros")

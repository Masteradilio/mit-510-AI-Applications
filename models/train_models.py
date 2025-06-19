#!/usr/bin/env python3
"""Script para treinar modelos de previsão de preços usando dados reais.
Treina modelos LSTM, GRU e SimpleRNN para múltiplos ativos.
Otimizado para GPU NVIDIA com CUDA e fallback para CPU.
Gera relatório de performance em markdown.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import warnings
import subprocess
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf

# Configurações
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
np.random.seed(42)
tf.random.set_seed(42)



# Configuração de CPU com limitação de núcleos
def setup_cpu():
    """Configura CPU com limitação de núcleos para não sobrecarregar o sistema."""
    print("🔧 Configurando dispositivo de processamento...")
    print("🖥️ Usando CPU com limitação de núcleos para preservar recursos do sistema...")
    
    import os
    total_cores = os.cpu_count()
    # Usar metade dos núcleos disponíveis (14 de 28)
    limited_cores = total_cores // 2
    
    print(f"💻 Total de núcleos disponíveis: {total_cores}")
    print(f"🎯 Limitando uso para: {limited_cores} núcleos")
    
    # Configurar TensorFlow para usar número limitado de threads
    tf.config.threading.set_intra_op_parallelism_threads(limited_cores)
    tf.config.threading.set_inter_op_parallelism_threads(limited_cores)
    
    print(f"✅ CPU configurada com {limited_cores} threads")
    return tf.distribute.get_strategy(), False
    
    try:
        # Método 1: PyNVML (NVIDIA Management Library)
        try:
            import pynvml
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if device_count > 0:
                gpu_available = True
                gpu_detected_by.append("PyNVML")
                print(f"✅ PyNVML detectou {device_count} GPU(s)")
                
                for i in range(device_count):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                    memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_info.append(name)
                    print(f"   GPU {i}: {name}")
                    print(f"   Memória: {memory_info.total // 1024**2} MB total, {memory_info.free // 1024**2} MB livre")
            pynvml.nvmlShutdown()
        except Exception as e:
            print(f"⚠️ PyNVML não disponível: {type(e).__name__}")
        
        # Método 2: Py3NVML (alternativa)
        try:
            import py3nvml.py3nvml as nvml
            nvml.nvmlInit()
            device_count = nvml.nvmlDeviceGetCount()
            if device_count > 0 and "PyNVML" not in gpu_detected_by:
                gpu_available = True
                gpu_detected_by.append("Py3NVML")
                print(f"✅ Py3NVML detectou {device_count} GPU(s)")
            nvml.nvmlShutdown()
        except Exception as e:
            print(f"⚠️ Py3NVML não disponível: {type(e).__name__}")
        
        # Método 3: GPUStat
        try:
            import gpustat
            gpu_stats = gpustat.GPUStatCollection.new_query()
            if len(gpu_stats.gpus) > 0:
                if not gpu_available:
                    gpu_available = True
                gpu_detected_by.append("GPUStat")
                print(f"✅ GPUStat detectou {len(gpu_stats.gpus)} GPU(s)")
                for gpu in gpu_stats.gpus:
                    print(f"   GPU {gpu.index}: {gpu.name} - {gpu.memory_used}MB/{gpu.memory_total}MB")
        except Exception as e:
            print(f"⚠️ GPUStat não disponível: {type(e).__name__}")
        
        # Método 4: TensorFlow GPU detection
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            if not gpu_available:
                gpu_available = True
            gpu_detected_by.append("TensorFlow")
            gpu_info.extend([gpu.name for gpu in gpus])
            print(f"✅ TensorFlow detectou {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
        
        # Método 5: CUDA availability check
        cuda_available = tf.test.is_built_with_cuda()
        print(f"🔧 TensorFlow compilado com CUDA: {cuda_available}")
        
        # Método 6: GPU compute capability
        if gpus:
            for i, gpu in enumerate(gpus):
                try:
                    gpu_details = tf.config.experimental.get_device_details(gpu)
                    compute_capability = gpu_details.get('compute_capability', 'N/A')
                    print(f"   GPU {i}: {gpu.name} - Compute Capability: {compute_capability}")
                except Exception as e:
                    print(f"   GPU {i}: {gpu.name} - Detalhes não disponíveis")
        
        # Método 7: Verificar drivers NVIDIA
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("✅ NVIDIA-SMI disponível - Drivers NVIDIA instalados")
                # Extrair informações básicas
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'RTX' in line or 'GeForce' in line or 'NVIDIA' in line:
                        print(f"   Detectado: {line.strip()}")
                        break
            else:
                print("⚠️ NVIDIA-SMI não disponível")
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            print(f"⚠️ Não foi possível executar nvidia-smi: {type(e).__name__}")
        
        # Método 8: Numba CUDA
        try:
            from numba import cuda
            if cuda.is_available():
                if not gpu_available:
                    gpu_available = True
                gpu_detected_by.append("Numba")
                print(f"✅ Numba CUDA disponível - {len(cuda.gpus)} GPU(s) detectada(s)")
                for i, gpu in enumerate(cuda.gpus):
                    print(f"   GPU {i}: {gpu.name} (CC {gpu.compute_capability})")
            else:
                print("⚠️ Numba CUDA não disponível")
        except Exception as e:
            print(f"⚠️ Erro ao verificar Numba CUDA: {e}")
        
        # Resumo da detecção
        if gpu_available:
            print(f"🎯 GPU detectada por: {', '.join(gpu_detected_by)}")
        
        if gpu_available and gpus:
            try:
                # Configurações otimizadas para RTX 5070Ti
                print("🚀 Configurando GPU para máxima performance...")
                
                for gpu in gpus:
                    # Habilitar crescimento de memória
                    tf.config.experimental.set_memory_growth(gpu, True)
                    
                    # Configurar limite de memória se necessário (RTX 5070Ti tem 16GB)
                    # tf.config.experimental.set_memory_limit(gpu, 14000)  # 14GB, deixando margem
                
                # Configurar estratégia de distribuição otimizada
                strategy = tf.distribute.MirroredStrategy(
                    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
                )
                
                # Verificar se a GPU está realmente sendo usada
                with strategy.scope():
                    test_tensor = tf.constant([1.0, 2.0, 3.0])
                    device_name = test_tensor.device
                    print(f"✅ Tensor de teste criado em: {device_name}")
                
                print(f"🎯 GPU configurada com sucesso: {len(gpus)} dispositivo(s)")
                print(f"   Estratégia: {type(strategy).__name__}")
                return strategy, True
                
            except Exception as e:
                print(f"❌ Erro ao configurar GPU: {e}")
                print("🔄 Fallback para CPU...")
                return tf.distribute.get_strategy(), False
        else:
            print("❌ Nenhuma GPU compatível detectada pelo TensorFlow")
            if gpu_available:
                print("💡 GPU detectada por outras bibliotecas, mas TensorFlow não consegue usá-la")
                print("💡 Considere reinstalar TensorFlow com suporte CUDA: pip install tensorflow[and-cuda]")
            print("🔄 Usando CPU")
            return tf.distribute.get_strategy(), False
            
    except Exception as e:
        print(f"❌ Erro na detecção de GPU: {e}")
        print("🔄 Usando CPU como fallback")
        return tf.distribute.get_strategy(), False



# Configurar dispositivo
strategy, using_gpu = setup_cpu()

# Configurações do projeto
PROJECT_ROOT = "C:/Users/Adilio/Documents/Projetos/MIT-510"
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

# Criar diretórios se não existirem
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Configurações de treinamento otimizadas
SEQUENCE_LENGTH = 60  # Usar 60 dias para prever o próximo
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Configurações dinâmicas baseadas na detecção de GPU
if using_gpu:
    EPOCHS = 100  # Épocas completas para GPU
    BATCH_SIZE = 128  # Batch size otimizado para RTX 5070Ti
    PATIENCE = 15  # Mais paciência com GPU
    print(f"🚀 Configuração GPU: {EPOCHS} épocas, batch size {BATCH_SIZE}")
else:
    EPOCHS = 50  # Épocas reduzidas para CPU
    BATCH_SIZE = 32  # Batch size menor para CPU
    PATIENCE = 10  # Menos paciência com CPU
    print(f"🖥️ Configuração CPU: {EPOCHS} épocas, batch size {BATCH_SIZE}")

# Mapeamento de arquivos de dados - Projeto focado em Apple e Bitcoin
DATA_FILES = {
    'aapl': 'apple-stock-2014-2024/apple-stockprice-2014-2024.csv',
    'btc': 'BTC-USD From 2014 To Dec-2024.csv'
}

class ModelTrainer:
    def __init__(self, strategy, using_gpu=False):
        self.results = []
        self.training_log = []
        self.strategy = strategy
        self.using_gpu = using_gpu
        print(f"🤖 ModelTrainer inicializado - GPU: {'Sim' if using_gpu else 'Não'}")
        
    def load_and_preprocess_data(self, asset_name):
        """Carrega e preprocessa os dados para um ativo específico."""
        file_path = os.path.join(DATA_RAW_DIR, DATA_FILES[asset_name])
        
        if not os.path.exists(file_path):
            print(f"❌ Arquivo não encontrado: {file_path}")
            return None, None, None
            
        print(f"📊 Carregando dados para {asset_name.upper()}...")
        
        # Carregar dados
        df = pd.read_csv(file_path)
        
        # Padronizar colunas
        if 'Adj Close' in df.columns:
            # Dados do Apple (tem Adj Close)
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()
            df['Close'] = df['Adj Close']  # Usar preço ajustado
        else:
            # Dados de crypto (não tem Adj Close)
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
            
        # Converter data
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Remover valores nulos
        df = df.dropna()
        
        # Criar features técnicas
        df['Price_Change'] = df['Close'].pct_change()
        df['Volatility'] = df['Price_Change'].rolling(window=10).std()
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
        
        # Remover linhas com NaN após criar features
        df = df.dropna()
        
        if len(df) < SEQUENCE_LENGTH + 100:
            print(f"❌ Dados insuficientes para {asset_name}: {len(df)} linhas")
            return None, None, None
            
        print(f"✅ Dados carregados: {len(df)} linhas de {df['Date'].min()} a {df['Date'].max()}")
        
        # Selecionar features para treinamento
        feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA_5', 'MA_20', 'Volume_MA']
        data = df[feature_columns].values
        
        # Normalizar dados
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        return scaled_data, scaler, df
    
    def create_sequences(self, data, sequence_length):
        """Cria sequências para treinamento."""
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i, 3])  # Índice 3 é o Close price
            
        return np.array(X), np.array(y)
    
    def create_model(self, model_type, input_shape):
        """Cria modelo baseado no tipo especificado com otimizações para GPU."""
        with self.strategy.scope():
            model = Sequential()
            
            # Unidades otimizadas para GPU
            units_1 = 128 if self.using_gpu else 100
            units_2 = 64 if self.using_gpu else 50
            units_3 = 32 if self.using_gpu else 25
            
            if model_type == 'lstm':
                model.add(LSTM(units_1, return_sequences=True, input_shape=input_shape))
                model.add(Dropout(0.2))
                model.add(LSTM(units_2, return_sequences=True))
                model.add(Dropout(0.2))
                model.add(LSTM(units_3, return_sequences=False))
                model.add(Dropout(0.2))
                
            elif model_type == 'gru':
                model.add(GRU(units_1, return_sequences=True, input_shape=input_shape))
                model.add(Dropout(0.2))
                model.add(GRU(units_2, return_sequences=True))
                model.add(Dropout(0.2))
                model.add(GRU(units_3, return_sequences=False))
                model.add(Dropout(0.2))
                
            elif model_type == 'simplernn':
                model.add(SimpleRNN(units_1, return_sequences=True, input_shape=input_shape))
                model.add(Dropout(0.2))
                model.add(SimpleRNN(units_2, return_sequences=True))
                model.add(Dropout(0.2))
                model.add(SimpleRNN(units_3, return_sequences=False))
                model.add(Dropout(0.2))
            
            model.add(Dense(units_3))
            model.add(Dense(1))
            
            # Compilar modelo com otimizações
            learning_rate = 0.002 if self.using_gpu else 0.001
            model.compile(
                optimizer=Adam(learning_rate=learning_rate),
                loss='mse',
                metrics=['mae']
            )
            
        return model
    
    def train_model(self, asset_name, model_type):
        """Treina um modelo específico para um ativo."""
        print(f"\n🚀 Iniciando treinamento: {asset_name.upper()} - {model_type.upper()}")
        
        # Carregar dados
        scaled_data, scaler, df = self.load_and_preprocess_data(asset_name)
        if scaled_data is None:
            return None
            
        # Criar sequências
        X, y = self.create_sequences(scaled_data, SEQUENCE_LENGTH)
        
        # Dividir dados
        train_size = int(len(X) * (1 - TEST_SIZE))
        val_size = int(train_size * (1 - VALIDATION_SIZE))
        
        X_train = X[:val_size]
        y_train = y[:val_size]
        X_val = X[val_size:train_size]
        y_val = y[val_size:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        print(f"📈 Dados divididos: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        # Criar modelo
        model = self.create_model(model_type, (X_train.shape[1], X_train.shape[2]))
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=1
        )
        
        # Treinar modelo com estratégia de distribuição
        device_info = "GPU" if self.using_gpu else "CPU"
        print(f"🔄 Treinando modelo {model_type.upper()} em {device_info}...")
        
        with self.strategy.scope():
            history = model.fit(
                X_train, y_train,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_data=(X_val, y_val),
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
        
        # Fazer previsões
        train_pred = model.predict(X_train, verbose=0)
        val_pred = model.predict(X_val, verbose=0)
        test_pred = model.predict(X_test, verbose=0)
        
        # Calcular métricas
        train_mse = mean_squared_error(y_train, train_pred)
        val_mse = mean_squared_error(y_val, val_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        
        train_mae = mean_absolute_error(y_train, train_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        train_r2 = r2_score(y_train, train_pred)
        val_r2 = r2_score(y_val, val_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        # Salvar modelo e scaler
        model_path = os.path.join(MODELS_DIR, f"{asset_name}_{model_type}_best.h5")
        scaler_path = os.path.join(MODELS_DIR, f"{asset_name}_scaler.joblib")
        
        model.save(model_path)
        joblib.dump(scaler, scaler_path)
        
        # Armazenar resultados
        result = {
            'asset': asset_name.upper(),
            'model_type': model_type.upper(),
            'train_mse': train_mse,
            'val_mse': val_mse,
            'test_mse': test_mse,
            'train_mae': train_mae,
            'val_mae': val_mae,
            'test_mae': test_mae,
            'train_r2': train_r2,
            'val_r2': val_r2,
            'test_r2': test_r2,
            'epochs_trained': len(history.history['loss']),
            'final_train_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'data_points': len(df),
            'training_period': f"{df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}"
        }
        
        self.results.append(result)
        
        print(f"✅ Modelo treinado com sucesso!")
        print(f"   📊 MSE Test: {test_mse:.6f}")
        print(f"   📊 MAE Test: {test_mae:.6f}")
        print(f"   📊 R² Test: {test_r2:.4f}")
        print(f"   💾 Modelo salvo: {model_path}")
        print(f"   💾 Scaler salvo: {scaler_path}")
        
        return result
    
    def generate_report(self):
        """Gera relatório de performance em markdown."""
        if not self.results:
            print("❌ Nenhum resultado para gerar relatório")
            return
            
        report_path = os.path.join(REPORTS_DIR, "model_training_report.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Relatório de Treinamento de Modelos\n\n")
            f.write(f"**Data de Geração:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Resumo geral
            f.write("## 📊 Resumo Geral\n\n")
            f.write(f"- **Total de modelos treinados:** {len(self.results)}\n")
            f.write(f"- **Ativos:** {len(set(r['asset'] for r in self.results))}\n")
            f.write(f"- **Tipos de modelo:** {len(set(r['model_type'] for r in self.results))}\n\n")
            
            # Melhores modelos por métrica
            f.write("## 🏆 Melhores Modelos\n\n")
            
            best_mse = min(self.results, key=lambda x: x['test_mse'])
            best_mae = min(self.results, key=lambda x: x['test_mae'])
            best_r2 = max(self.results, key=lambda x: x['test_r2'])
            
            f.write(f"### Menor MSE (Test)\n")
            f.write(f"**{best_mse['asset']} - {best_mse['model_type']}:** {best_mse['test_mse']:.6f}\n\n")
            
            f.write(f"### Menor MAE (Test)\n")
            f.write(f"**{best_mae['asset']} - {best_mae['model_type']}:** {best_mae['test_mae']:.6f}\n\n")
            
            f.write(f"### Maior R² (Test)\n")
            f.write(f"**{best_r2['asset']} - {best_r2['model_type']}:** {best_r2['test_r2']:.4f}\n\n")
            
            # Tabela detalhada
            f.write("## 📋 Resultados Detalhados\n\n")
            f.write("| Ativo | Modelo | MSE Test | MAE Test | R² Test | Épocas | Período de Dados |\n")
            f.write("|-------|--------|----------|----------|---------|--------|------------------|\n")
            
            for result in sorted(self.results, key=lambda x: x['test_mse']):
                f.write(f"| {result['asset']} | {result['model_type']} | "
                       f"{result['test_mse']:.6f} | {result['test_mae']:.6f} | "
                       f"{result['test_r2']:.4f} | {result['epochs_trained']} | "
                       f"{result['training_period']} |\n")
            
            # Análise por ativo
            f.write("\n## 📈 Análise por Ativo\n\n")
            
            assets = set(r['asset'] for r in self.results)
            for asset in sorted(assets):
                asset_results = [r for r in self.results if r['asset'] == asset]
                f.write(f"### {asset}\n\n")
                
                best_asset_model = min(asset_results, key=lambda x: x['test_mse'])
                f.write(f"**Melhor modelo:** {best_asset_model['model_type']} "
                       f"(MSE: {best_asset_model['test_mse']:.6f})\n\n")
                
                for result in asset_results:
                    f.write(f"- **{result['model_type']}:** "
                           f"MSE={result['test_mse']:.6f}, "
                           f"MAE={result['test_mae']:.6f}, "
                           f"R²={result['test_r2']:.4f}\n")
                f.write("\n")
            
            # Análise por tipo de modelo
            f.write("## 🤖 Análise por Tipo de Modelo\n\n")
            
            model_types = set(r['model_type'] for r in self.results)
            for model_type in sorted(model_types):
                type_results = [r for r in self.results if r['model_type'] == model_type]
                avg_mse = np.mean([r['test_mse'] for r in type_results])
                avg_mae = np.mean([r['test_mae'] for r in type_results])
                avg_r2 = np.mean([r['test_r2'] for r in type_results])
                
                f.write(f"### {model_type}\n\n")
                f.write(f"- **MSE médio:** {avg_mse:.6f}\n")
                f.write(f"- **MAE médio:** {avg_mae:.6f}\n")
                f.write(f"- **R² médio:** {avg_r2:.4f}\n")
                f.write(f"- **Modelos treinados:** {len(type_results)}\n\n")
            
            # Configurações utilizadas
            f.write("## ⚙️ Configurações de Treinamento\n\n")
            f.write(f"- **Sequência de entrada:** {SEQUENCE_LENGTH} dias\n")
            f.write(f"- **Tamanho do teste:** {TEST_SIZE*100}%\n")
            f.write(f"- **Tamanho da validação:** {VALIDATION_SIZE*100}%\n")
            f.write(f"- **Épocas máximas:** {EPOCHS}\n")
            f.write(f"- **Batch size:** {BATCH_SIZE}\n")
            f.write(f"- **Paciência (early stopping):** {PATIENCE}\n")
            f.write(f"- **Dispositivo:** {'GPU' if using_gpu else 'CPU'}\n\n")
            
            # Conclusões
            f.write("## 🎯 Conclusões\n\n")
            
            avg_mse = np.mean([r['test_mse'] for r in self.results])
            models_low_mse = len([r for r in self.results if r['test_mse'] < 0.01])
            
            f.write(f"- **MSE médio geral:** {avg_mse:.6f}\n")
            f.write(f"- **Modelos com MSE < 0.01:** {models_low_mse}/{len(self.results)}\n")
            
            if avg_mse < 0.01:
                f.write("- ✅ **Status:** Modelos com performance satisfatória (MSE baixo)\n")
            else:
                f.write("- ⚠️ **Status:** Alguns modelos podem precisar de ajustes\n")
                
            f.write("\n---\n")
            f.write(f"*Relatório gerado automaticamente pelo sistema de treinamento de modelos.*\n")
            f.write(f"*Dispositivo utilizado: {'GPU NVIDIA RTX 5070Ti' if using_gpu else 'CPU'}*\n")
        
        print(f"📄 Relatório gerado: {report_path}")
        return report_path

def main():
    """Função principal para treinar todos os modelos."""
    print("🚀 Iniciando treinamento de modelos de previsão de preços")
    print("="*60)
    
    trainer = ModelTrainer(strategy, using_gpu)
    model_types = ['lstm', 'gru', 'simplernn']
    
    # Treinar modelos para cada ativo e tipo
    total_models = len(DATA_FILES) * len(model_types)
    current_model = 0
    
    for asset_name in DATA_FILES.keys():
        for model_type in model_types:
            current_model += 1
            print(f"\n📊 Progresso: {current_model}/{total_models}")
            
            try:
                result = trainer.train_model(asset_name, model_type)
                if result is None:
                    print(f"❌ Falha no treinamento: {asset_name} - {model_type}")
            except Exception as e:
                print(f"❌ Erro no treinamento {asset_name} - {model_type}: {str(e)}")
                continue
    
    # Gerar relatório
    print("\n" + "="*60)
    print("📄 Gerando relatório de performance...")
    
    if trainer.results:
        report_path = trainer.generate_report()
        
        # Verificar se todos os modelos têm MSE baixo
        avg_mse = np.mean([r['test_mse'] for r in trainer.results])
        high_mse_models = [r for r in trainer.results if r['test_mse'] > 0.01]
        
        print("\n" + "="*60)
        print("🎯 RESUMO FINAL")
        print("="*60)
        print(f"✅ Modelos treinados com sucesso: {len(trainer.results)}")
        print(f"📊 MSE médio: {avg_mse:.6f}")
        
        if high_mse_models:
            print(f"⚠️  Modelos com MSE alto (>0.01): {len(high_mse_models)}")
            for model in high_mse_models:
                print(f"   - {model['asset']} {model['model_type']}: {model['test_mse']:.6f}")
        else:
            print("✅ Todos os modelos têm MSE satisfatório (<0.01)")
            
        print(f"\n📄 Relatório completo: {report_path}")
        print("\n🎉 Treinamento concluído com sucesso!")
        
    else:
        print("❌ Nenhum modelo foi treinado com sucesso")
        sys.exit(1)

if __name__ == "__main__":
    main()
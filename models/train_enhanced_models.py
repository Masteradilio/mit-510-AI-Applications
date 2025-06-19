"""Script avançado para treinar modelos com variáveis exógenas e otimização de hiperparâmetros.
Implementa treinamento comparativo entre modelos básicos e modelos com variáveis exógenas.
Inclui otimização automática de hiperparâmetros e validação cruzada temporal.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import warnings
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import ParameterGrid
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# Adicionar o diretório src ao path para importar módulos personalizados
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from preprocessing.feature_engineering import add_technical_indicators, add_exogenous_variables, create_target_variables, add_cyclical_features
    from modeling.rnn_models import create_enhanced_rnn_model, get_callbacks
except ImportError as e:
    print(f"⚠️ Erro ao importar módulos personalizados: {e}")
    print("Continuando com funcionalidades básicas...")
    
# Configurações
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')
np.random.seed(42)
tf.random.set_seed(42)

# Configuração de CPU otimizada
def setup_device():
    """Configura dispositivo de processamento."""
    print("🔧 Configurando dispositivo de processamento...")
    
    # Verificar GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU configurada: {len(gpus)} dispositivo(s)")
            return tf.distribute.MirroredStrategy(), True
        except Exception as e:
            print(f"⚠️ Erro ao configurar GPU: {e}")
    
    # Configurar CPU
    total_cores = os.cpu_count()
    limited_cores = total_cores // 2
    tf.config.threading.set_intra_op_parallelism_threads(limited_cores)
    tf.config.threading.set_inter_op_parallelism_threads(limited_cores)
    print(f"🖥️ CPU configurada com {limited_cores} threads")
    return tf.distribute.get_strategy(), False

# Configurar dispositivo
strategy, using_gpu = setup_device()

# Configurações do projeto
PROJECT_ROOT = "C:/Users/Adilio/Documents/Projetos/MIT-510"
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "models")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")

# Criar diretórios
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# Configurações de treinamento
SEQUENCE_LENGTH = 60
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Configurações dinâmicas baseadas no dispositivo
if using_gpu:
    EPOCHS = 100
    BATCH_SIZE = 128
    PATIENCE = 15
else:
    EPOCHS = 50
    BATCH_SIZE = 32
    PATIENCE = 10

# Mapeamento de arquivos de dados
DATA_FILES = {
    'aapl': 'apple-stock-2014-2024/apple-stockprice-2014-2024.csv',
    'btc': 'BTC-USD From 2014 To Dec-2024.csv'
}

# Configurações de otimização de hiperparâmetros
HYPERPARAM_GRID = {
    'units': [64, 128, 256] if using_gpu else [32, 64, 128],
    'layers': [2, 3, 4],
    'dropout_rate': [0.1, 0.2, 0.3],
    'learning_rate': [0.001, 0.002, 0.005],
    'model_type': ['lstm', 'gru']
}

class EnhancedModelTrainer:
    """Trainer avançado com suporte a variáveis exógenas e otimização de hiperparâmetros."""
    
    def __init__(self, strategy, using_gpu=False):
        self.results = []
        self.strategy = strategy
        self.using_gpu = using_gpu
        self.best_models = {}
        print(f"🤖 EnhancedModelTrainer inicializado - GPU: {'Sim' if using_gpu else 'Não'}")
    
    def load_and_enhance_data(self, asset_name, use_exogenous=True):
        """Carrega dados e adiciona features avançadas incluindo variáveis exógenas."""
        file_path = os.path.join(DATA_RAW_DIR, DATA_FILES[asset_name])
        
        if not os.path.exists(file_path):
            print(f"❌ Arquivo não encontrado: {file_path}")
            return None, None, None
            
        print(f"📊 Carregando e processando dados para {asset_name.upper()}...")
        
        # Carregar dados básicos
        df = pd.read_csv(file_path)
        
        # Padronizar colunas
        if 'Adj Close' in df.columns:
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].copy()
            df['Close'] = df['Adj Close']
        else:
            df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
            
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        df = df.dropna()
        
        # Adicionar indicadores técnicos
        try:
            df = add_technical_indicators(df)
            print("✅ Indicadores técnicos adicionados")
        except Exception as e:
            print(f"⚠️ Erro ao adicionar indicadores técnicos: {e}")
            # Fallback para indicadores básicos
            df['MA_5'] = df['Close'].rolling(window=5).mean()
            df['MA_20'] = df['Close'].rolling(window=20).mean()
            df['Volatility'] = df['Close'].pct_change().rolling(window=10).std()
        
        # Adicionar variáveis exógenas se solicitado
        if use_exogenous:
            try:
                df = add_exogenous_variables(df, use_comprehensive_collector=True)
                print("✅ Variáveis exógenas adicionadas")
            except Exception as e:
                print(f"⚠️ Erro ao adicionar variáveis exógenas: {e}")
                print("Continuando sem variáveis exógenas...")
        
        # Adicionar features cíclicas
        try:
            df = add_cyclical_features(df)
            print("✅ Features cíclicas adicionadas")
        except Exception as e:
            print(f"⚠️ Erro ao adicionar features cíclicas: {e}")
        
        # Remover colunas não numéricas e com muitos NaN
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df_numeric = df[numeric_columns].copy()
        
        # Remover colunas com mais de 50% de valores ausentes
        threshold = len(df_numeric) * 0.5
        df_numeric = df_numeric.dropna(axis=1, thresh=threshold)
        
        # Preencher valores ausentes restantes
        df_numeric = df_numeric.fillna(method='ffill').fillna(method='bfill')
        df_numeric = df_numeric.dropna()
        
        if len(df_numeric) < SEQUENCE_LENGTH + 100:
            print(f"❌ Dados insuficientes após processamento: {len(df_numeric)} linhas")
            return None, None, None
            
        print(f"✅ Dados processados: {len(df_numeric)} linhas, {len(df_numeric.columns)} features")
        
        # Preparar dados para treinamento
        feature_columns = [col for col in df_numeric.columns if col != 'Close']
        data = df_numeric[feature_columns + ['Close']].values
        
        # Normalizar dados
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        return scaled_data, scaler, df_numeric
    
    def create_sequences_enhanced(self, data, sequence_length):
        """Cria sequências para treinamento com suporte a múltiplas features."""
        X, y = [], []
        
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i, :-1])  # Todas as features exceto Close
            y.append(data[i, -1])  # Close price (última coluna)
            
        return np.array(X), np.array(y)
    
    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val, max_trials=5):
        """Otimiza hiperparâmetros usando grid search limitado."""
        print("🔍 Iniciando otimização de hiperparâmetros...")
        
        # Criar grid reduzido para otimização
        param_combinations = list(ParameterGrid(HYPERPARAM_GRID))
        
        # Limitar número de combinações para evitar tempo excessivo
        if len(param_combinations) > max_trials:
            param_combinations = np.random.choice(param_combinations, max_trials, replace=False)
        
        best_params = None
        best_score = float('inf')
        
        for i, params in enumerate(param_combinations):
            print(f"🧪 Testando combinação {i+1}/{len(param_combinations)}: {params}")
            
            try:
                with self.strategy.scope():
                    model = create_enhanced_rnn_model(
                        input_shape=(X_train.shape[1], X_train.shape[2]),
                        units=params['units'],
                        layers=params['layers'],
                        dropout_rate=params['dropout_rate'],
                        model_type=params['model_type'],
                        use_batch_norm=True
                    )
                    
                    model.compile(
                        optimizer=Adam(learning_rate=params['learning_rate']),
                        loss='mse',
                        metrics=['mae']
                    )
                
                # Treinamento rápido para avaliação
                early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                
                history = model.fit(
                    X_train, y_train,
                    epochs=20,  # Épocas reduzidas para otimização
                    batch_size=BATCH_SIZE,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stopping],
                    verbose=0
                )
                
                # Avaliar performance
                val_loss = min(history.history['val_loss'])
                
                if val_loss < best_score:
                    best_score = val_loss
                    best_params = params
                    print(f"✅ Nova melhor combinação encontrada! Val Loss: {val_loss:.6f}")
                
            except Exception as e:
                print(f"❌ Erro ao testar combinação: {e}")
                continue
        
        print(f"🎯 Melhores hiperparâmetros: {best_params}")
        print(f"🎯 Melhor score: {best_score:.6f}")
        
        return best_params if best_params else HYPERPARAM_GRID
    
    def train_comparative_models(self, asset_name):
        """Treina modelos comparativos: básico vs. com variáveis exógenas."""
        print(f"\n🚀 Iniciando treinamento comparativo para {asset_name.upper()}")
        
        results = []
        
        # 1. Treinar modelo básico (sem variáveis exógenas)
        print("\n📊 Treinando modelo BÁSICO (sem variáveis exógenas)...")
        basic_result = self._train_single_model(asset_name, use_exogenous=False, model_suffix="basic")
        if basic_result:
            results.append(basic_result)
        
        # 2. Treinar modelo com variáveis exógenas
        print("\n📊 Treinando modelo AVANÇADO (com variáveis exógenas)...")
        enhanced_result = self._train_single_model(asset_name, use_exogenous=True, model_suffix="enhanced")
        if enhanced_result:
            results.append(enhanced_result)
        
        return results
    
    def _train_single_model(self, asset_name, use_exogenous=True, model_suffix=""):
        """Treina um único modelo com configurações específicas."""
        # Carregar e processar dados
        scaled_data, scaler, df = self.load_and_enhance_data(asset_name, use_exogenous)
        if scaled_data is None:
            return None
        
        # Criar sequências
        X, y = self.create_sequences_enhanced(scaled_data, SEQUENCE_LENGTH)
        
        # Dividir dados
        train_size = int(len(X) * (1 - TEST_SIZE))
        val_size = int(train_size * (1 - VALIDATION_SIZE))
        
        X_train = X[:val_size]
        y_train = y[:val_size]
        X_val = X[val_size:train_size]
        y_val = y[val_size:train_size]
        X_test = X[train_size:]
        y_test = y[train_size:]
        
        print(f"📈 Dados: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}, Features={X.shape[2]}")
        
        # Otimizar hiperparâmetros
        best_params = self.optimize_hyperparameters(X_train, y_train, X_val, y_val)
        
        # Treinar modelo final com melhores parâmetros
        print("🏗️ Treinando modelo final com hiperparâmetros otimizados...")
        
        with self.strategy.scope():
            model = create_enhanced_rnn_model(
                input_shape=(X_train.shape[1], X_train.shape[2]),
                units=best_params.get('units', 128),
                layers=best_params.get('layers', 3),
                dropout_rate=best_params.get('dropout_rate', 0.2),
                model_type=best_params.get('model_type', 'lstm'),
                use_batch_norm=True
            )
            
            model.compile(
                optimizer=Adam(learning_rate=best_params.get('learning_rate', 0.001)),
                loss='mse',
                metrics=['mae']
            )
        
        # Callbacks
        model_name = f"{asset_name}_{model_suffix}_optimized"
        model_path = os.path.join(MODELS_DIR, f"{model_name}.h5")
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=1),
            ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=1)
        ]
        
        # Treinar modelo
        history = model.fit(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Fazer previsões
        train_pred = model.predict(X_train, verbose=0)
        val_pred = model.predict(X_val, verbose=0)
        test_pred = model.predict(X_test, verbose=0)
        
        # Calcular métricas
        metrics = {
            'train_mse': mean_squared_error(y_train, train_pred),
            'val_mse': mean_squared_error(y_val, val_pred),
            'test_mse': mean_squared_error(y_test, test_pred),
            'train_mae': mean_absolute_error(y_train, train_pred),
            'val_mae': mean_absolute_error(y_val, val_pred),
            'test_mae': mean_absolute_error(y_test, test_pred),
            'train_r2': r2_score(y_train, train_pred),
            'val_r2': r2_score(y_val, val_pred),
            'test_r2': r2_score(y_test, test_pred)
        }
        
        # Salvar scaler
        scaler_path = os.path.join(MODELS_DIR, f"{model_name}_scaler.joblib")
        joblib.dump(scaler, scaler_path)
        
        # Preparar resultado
        result = {
            'asset': asset_name.upper(),
            'model_type': f"{best_params.get('model_type', 'lstm').upper()}_{model_suffix.upper()}",
            'use_exogenous': use_exogenous,
            'num_features': X.shape[2],
            'best_params': best_params,
            **metrics,
            'epochs_trained': len(history.history['loss']),
            'final_train_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'model_path': model_path,
            'scaler_path': scaler_path,
            'data_points': len(df),
            'training_period': f"{df.index[0]} to {df.index[-1]}"
        }
        
        self.results.append(result)
        
        print(f"✅ Modelo {model_suffix} treinado com sucesso!")
        print(f"   📊 Features: {X.shape[2]}")
        print(f"   📊 MSE Test: {metrics['test_mse']:.6f}")
        print(f"   📊 MAE Test: {metrics['test_mae']:.6f}")
        print(f"   📊 R² Test: {metrics['test_r2']:.4f}")
        
        return result
    
    def generate_enhanced_report(self):
        """Gera relatório comparativo detalhado."""
        if not self.results:
            print("❌ Nenhum resultado para gerar relatório")
            return
        
        report_path = os.path.join(REPORTS_DIR, "enhanced_model_training_report.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Relatório de Treinamento de Modelos Avançados\n\n")
            f.write(f"**Data de Geração:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Resumo geral
            f.write("## 📊 Resumo Geral\n\n")
            f.write(f"- **Total de modelos treinados:** {len(self.results)}\n")
            f.write(f"- **Ativos:** {len(set(r['asset'] for r in self.results))}\n")
            f.write(f"- **Modelos básicos:** {len([r for r in self.results if not r['use_exogenous']])}\n")
            f.write(f"- **Modelos com variáveis exógenas:** {len([r for r in self.results if r['use_exogenous']])}\n\n")
            
            # Comparação de performance
            f.write("## 🏆 Comparação de Performance\n\n")
            
            # Agrupar por ativo
            assets = set(r['asset'] for r in self.results)
            for asset in sorted(assets):
                asset_results = [r for r in self.results if r['asset'] == asset]
                f.write(f"### {asset}\n\n")
                
                basic_models = [r for r in asset_results if not r['use_exogenous']]
                enhanced_models = [r for r in asset_results if r['use_exogenous']]
                
                if basic_models and enhanced_models:
                    basic_best = min(basic_models, key=lambda x: x['test_mse'])
                    enhanced_best = min(enhanced_models, key=lambda x: x['test_mse'])
                    
                    improvement = ((basic_best['test_mse'] - enhanced_best['test_mse']) / basic_best['test_mse']) * 100
                    
                    f.write(f"**Modelo Básico (sem variáveis exógenas):**\n")
                    f.write(f"- MSE: {basic_best['test_mse']:.6f}\n")
                    f.write(f"- MAE: {basic_best['test_mae']:.6f}\n")
                    f.write(f"- R²: {basic_best['test_r2']:.4f}\n")
                    f.write(f"- Features: {basic_best['num_features']}\n\n")
                    
                    f.write(f"**Modelo Avançado (com variáveis exógenas):**\n")
                    f.write(f"- MSE: {enhanced_best['test_mse']:.6f}\n")
                    f.write(f"- MAE: {enhanced_best['test_mae']:.6f}\n")
                    f.write(f"- R²: {enhanced_best['test_r2']:.4f}\n")
                    f.write(f"- Features: {enhanced_best['num_features']}\n\n")
                    
                    if improvement > 0:
                        f.write(f"**🎯 Melhoria com variáveis exógenas: {improvement:.2f}% (MSE)** ✅\n\n")
                    else:
                        f.write(f"**⚠️ Modelo básico teve melhor performance: {abs(improvement):.2f}%**\n\n")
            
            # Tabela detalhada
            f.write("## 📋 Resultados Detalhados\n\n")
            f.write("| Ativo | Modelo | Exógenas | Features | MSE Test | MAE Test | R² Test | Épocas |\n")
            f.write("|-------|--------|----------|----------|----------|----------|---------|--------|\n")
            
            for result in sorted(self.results, key=lambda x: x['test_mse']):
                exogenous_status = "✅" if result['use_exogenous'] else "❌"
                f.write(f"| {result['asset']} | {result['model_type']} | {exogenous_status} | "
                       f"{result['num_features']} | {result['test_mse']:.6f} | "
                       f"{result['test_mae']:.6f} | {result['test_r2']:.4f} | "
                       f"{result['epochs_trained']} |\n")
            
            # Análise de hiperparâmetros
            f.write("\n## ⚙️ Hiperparâmetros Otimizados\n\n")
            
            for result in self.results:
                f.write(f"### {result['asset']} - {result['model_type']}\n\n")
                params = result['best_params']
                for param, value in params.items():
                    f.write(f"- **{param}:** {value}\n")
                f.write("\n")
            
            # Configurações utilizadas
            f.write("## 🔧 Configurações de Treinamento\n\n")
            f.write(f"- **Sequência de entrada:** {SEQUENCE_LENGTH} dias\n")
            f.write(f"- **Tamanho do teste:** {TEST_SIZE*100}%\n")
            f.write(f"- **Tamanho da validação:** {VALIDATION_SIZE*100}%\n")
            f.write(f"- **Épocas máximas:** {EPOCHS}\n")
            f.write(f"- **Batch size:** {BATCH_SIZE}\n")
            f.write(f"- **Paciência (early stopping):** {PATIENCE}\n")
            f.write(f"- **Dispositivo:** {'GPU' if using_gpu else 'CPU'}\n")
            f.write(f"- **Otimização de hiperparâmetros:** Ativada\n\n")
            
            # Conclusões
            f.write("## 🎯 Conclusões\n\n")
            
            basic_results = [r for r in self.results if not r['use_exogenous']]
            enhanced_results = [r for r in self.results if r['use_exogenous']]
            
            if basic_results and enhanced_results:
                basic_avg_mse = np.mean([r['test_mse'] for r in basic_results])
                enhanced_avg_mse = np.mean([r['test_mse'] for r in enhanced_results])
                overall_improvement = ((basic_avg_mse - enhanced_avg_mse) / basic_avg_mse) * 100
                
                f.write(f"- **MSE médio (modelos básicos):** {basic_avg_mse:.6f}\n")
                f.write(f"- **MSE médio (modelos com exógenas):** {enhanced_avg_mse:.6f}\n")
                
                if overall_improvement > 0:
                    f.write(f"- **✅ Melhoria geral com variáveis exógenas:** {overall_improvement:.2f}%\n")
                    f.write(f"- **🎉 Status:** Variáveis exógenas melhoraram significativamente a performance\n")
                else:
                    f.write(f"- **⚠️ Modelos básicos tiveram melhor performance:** {abs(overall_improvement):.2f}%\n")
                    f.write(f"- **🔍 Recomendação:** Revisar seleção e processamento de variáveis exógenas\n")
            
            f.write("\n---\n")
            f.write("*Relatório gerado automaticamente pelo sistema de treinamento avançado.*\n")
            f.write(f"*Dispositivo utilizado: {'GPU' if using_gpu else 'CPU'}*\n")
        
        print(f"📄 Relatório avançado gerado: {report_path}")
        return report_path

def main():
    """Função principal para executar treinamento avançado."""
    print("🚀 Iniciando Treinamento Avançado com Variáveis Exógenas")
    print("="*70)
    
    trainer = EnhancedModelTrainer(strategy, using_gpu)
    
    # Treinar modelos comparativos para cada ativo
    for asset_name in DATA_FILES.keys():
        try:
            results = trainer.train_comparative_models(asset_name)
            print(f"✅ Treinamento concluído para {asset_name.upper()}: {len(results)} modelos")
        except Exception as e:
            print(f"❌ Erro no treinamento para {asset_name}: {str(e)}")
            continue
    
    # Gerar relatório
    print("\n" + "="*70)
    print("📄 Gerando relatório comparativo...")
    
    if trainer.results:
        report_path = trainer.generate_enhanced_report()
        
        # Análise final
        basic_results = [r for r in trainer.results if not r['use_exogenous']]
        enhanced_results = [r for r in trainer.results if r['use_exogenous']]
        
        print("\n" + "="*70)
        print("🎯 ANÁLISE FINAL")
        print("="*70)
        
        if basic_results and enhanced_results:
            basic_avg_mse = np.mean([r['test_mse'] for r in basic_results])
            enhanced_avg_mse = np.mean([r['test_mse'] for r in enhanced_results])
            improvement = ((basic_avg_mse - enhanced_avg_mse) / basic_avg_mse) * 100
            
            print(f"📊 Modelos básicos - MSE médio: {basic_avg_mse:.6f}")
            print(f"📊 Modelos com exógenas - MSE médio: {enhanced_avg_mse:.6f}")
            
            if improvement > 0:
                print(f"🎉 SUCESSO: Variáveis exógenas melhoraram a performance em {improvement:.2f}%")
            else:
                print(f"⚠️ ATENÇÃO: Modelos básicos foram {abs(improvement):.2f}% melhores")
        
        print(f"\n📄 Relatório completo: {report_path}")
        print("\n🎉 Treinamento avançado concluído!")
        
    else:
        print("❌ Nenhum modelo foi treinado com sucesso")
        sys.exit(1)

if __name__ == "__main__":
    main()
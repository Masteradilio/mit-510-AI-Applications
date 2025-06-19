# -*- coding: utf-8 -*-
"""Script para executar todos os testes unitários e de integração."""

import os
import sys
import pytest
import unittest
import subprocess
from pathlib import Path

# Adicionar diretório raiz ao path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

def run_tests():
    """Executa todos os testes unitários e de integração."""
    print("=" * 80)
    print("INICIANDO TESTES DO PROJETO AGILE CAPITAL FORECAST")
    print("=" * 80)
    
    # Verificar estrutura de diretórios
    print("\n[1/5] Verificando estrutura de diretórios...")
    required_dirs = [
        "data/raw",
        "data/processed",
        "notebooks",
        "src/data_ingestion",
        "src/preprocessing",
        "src/modeling",
        "src/app/components",
        "src/app/pages",
        "tests/unit/modeling",
        "tests/unit/app/components",
        "tests/integration",
        "models",
        "reports"
    ]
    
    for dir_path in required_dirs:
        full_path = os.path.join(project_root, dir_path)
        if not os.path.exists(full_path):
            print(f"  CRIANDO: {dir_path}")
            os.makedirs(full_path, exist_ok=True)
        else:
            print(f"  OK: {dir_path}")
    
    # Verificar arquivos principais
    print("\n[2/5] Verificando arquivos principais...")
    required_files = [
        "src/data_ingestion/loader.py",
        "src/preprocessing/feature_engineering.py",
        "src/preprocessing/scalers_transformers.py",
        "src/modeling/rnn_models.py",
        "src/modeling/prediction.py",
        "src/modeling/strategy_simulation.py",
        "src/app/components/plotting.py",
        "src/app/components/ui_elements.py",
        "notebooks/financial_forecasting_eda_modeling.ipynb",
        "app.py"
    ]
    
    for file_path in required_files:
        full_path = os.path.join(project_root, file_path)
        if os.path.exists(full_path):
            print(f"  OK: {file_path}")
        else:
            print(f"  FALTANDO: {file_path}")
    
    # Executar testes unitários
    print("\n[3/5] Executando testes unitários...")
    unit_test_dirs = [
        "tests/unit/modeling",
        "tests/unit/app/components"
    ]
    
    for test_dir in unit_test_dirs:
        full_path = os.path.join(project_root, test_dir)
        test_files = [f for f in os.listdir(full_path) if f.startswith("test_") and f.endswith(".py")]
        
        if test_files:
            print(f"\n  Executando testes em {test_dir}:")
            for test_file in test_files:
                test_path = os.path.join(full_path, test_file)
                print(f"    - {test_file}...")
                try:
                    result = subprocess.run(
                        [sys.executable, "-m", "unittest", test_path],
                        capture_output=True,
                        text=True
                    )
                    if result.returncode == 0:
                        print(f"      PASSOU")
                    else:
                        print(f"      FALHOU")
                        print(f"      {result.stderr}")
                except Exception as e:
                    print(f"      ERRO: {e}")
        else:
            print(f"  Nenhum arquivo de teste encontrado em {test_dir}")
    
    # Executar testes de integração
    print("\n[4/5] Executando testes de integração...")
    integration_test_dir = os.path.join(project_root, "tests/integration")
    integration_test_files = [f for f in os.listdir(integration_test_dir) if f.startswith("test_") and f.endswith(".py")]
    
    if integration_test_files:
        print(f"\n  Executando testes de integração:")
        for test_file in integration_test_files:
            test_path = os.path.join(integration_test_dir, test_file)
            print(f"    - {test_file}...")
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "unittest", test_path],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0:
                    print(f"      PASSOU")
                else:
                    print(f"      FALHOU")
                    print(f"      {result.stderr}")
            except Exception as e:
                print(f"      ERRO: {e}")
    else:
        print(f"  Nenhum arquivo de teste de integração encontrado")
    
    # Gerar relatório de cobertura
    print("\n[5/5] Gerando relatório de cobertura...")
    try:
        coverage_dir = os.path.join(project_root, "reports", "coverage")
        os.makedirs(coverage_dir, exist_ok=True)
        
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--cov=src", "--cov-report=html:reports/coverage", "tests/"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"  Relatório de cobertura gerado com sucesso em reports/coverage/")
        else:
            print(f"  Falha ao gerar relatório de cobertura")
            print(f"  {result.stderr}")
    except Exception as e:
        print(f"  ERRO ao gerar relatório de cobertura: {e}")
    
    print("\n" + "=" * 80)
    print("TESTES CONCLUÍDOS")
    print("=" * 80)

if __name__ == "__main__":
    run_tests()

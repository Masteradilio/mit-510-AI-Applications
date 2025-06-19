import streamlit as st
import os
from pathlib import Path

# Configuração da página
st.set_page_config(
    page_title="Documentação - MIT-510",
    page_icon="📚",
    layout="wide"
)

# Título da página
st.title("📚 Documentação do Projeto")
st.markdown("---")

# Função para carregar o README.md
@st.cache_data
def load_readme():
    """Carrega o conteúdo do arquivo README.md"""
    try:
        # Caminho para o README.md (relativo ao diretório do projeto)
        current_dir = Path(__file__).parent.parent.parent.parent
        readme_path = current_dir / "docs" / "README.md"
        
        if readme_path.exists():
            with open(readme_path, 'r', encoding='utf-8') as file:
                return file.read()
        else:
            return "❌ Arquivo README.md não encontrado."
    except Exception as e:
        return f"❌ Erro ao carregar o README.md: {str(e)}"

# Carregar e exibir o conteúdo do README
readme_content = load_readme()

# Verificar se o conteúdo foi carregado com sucesso
if readme_content.startswith("❌"):
    st.error(readme_content)
else:
    # Adicionar informações sobre a documentação
    st.info(
        "📖 **Sobre esta página**: Esta página exibe a documentação completa do projeto MIT-510, "
        "incluindo instruções de instalação, uso, arquitetura e todas as funcionalidades implementadas."
    )
    
    # Criar abas para organizar o conteúdo
    tab1, tab2 = st.tabs(["📖 Documentação Completa", "🔍 Navegação Rápida"])
    
    with tab1:
        # Exibir o conteúdo do README em markdown
        st.markdown(readme_content)
    
    with tab2:
        st.subheader("🧭 Navegação Rápida")
        
        # Criar links para seções importantes
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### 📋 Seções Principais
            - [Visão Geral](#visão-geral)
            - [Arquitetura do Sistema](#arquitetura-do-sistema)
            - [Configuração do Ambiente](#configuração-do-ambiente)
            - [Execução](#execução)
            - [Modelos RNN](#modelos-rnn)
            """)
        
        with col2:
            st.markdown("""
            ### 🔧 Recursos Técnicos
            - [Indicadores Técnicos](#indicadores-técnicos-implementados)
            - [Métricas de Avaliação](#métricas-de-avaliação-avançadas)
            - [Estratégias de Trading](#estratégias-de-trading-avançadas)
            - [Desenvolvimento](#desenvolvimento-e-extensibilidade)
            - [Limitações](#limitações-e-considerações)
            """)
        
        # Informações sobre o status do projeto
        st.subheader("📊 Status Atual do Projeto")
        
        # Métricas do projeto
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Páginas do App", "5", "Completas")
        
        with col2:
            st.metric("Modelos RNN", "3", "LSTM, GRU, SimpleRNN")
        
        with col3:
            st.metric("Indicadores Técnicos", "10+", "Implementados")
        
        with col4:
            st.metric("Estratégias Trading", "5", "Com gestão de risco")
        
        # Progresso das funcionalidades
        st.subheader("🚀 Progresso das Funcionalidades")
        
        progress_data = {
            "Pipeline de Dados": 100,
            "Modelos RNN": 100,
            "Análise Exploratória": 100,
            "Estratégias de Trading": 100,
            "Interface Streamlit": 100,
            "Variáveis Exógenas": 75,
            "Relatório Final": 90
        }
        
        for feature, progress in progress_data.items():
            st.progress(progress/100, text=f"{feature}: {progress}%")

# Rodapé com informações adicionais
st.markdown("---")
st.markdown("""
### 📞 Suporte e Contato

- **Projeto**: MIT-510 - Previsão de Ativos Financeiros com IA
- **Tipo**: Projeto Acadêmico
- **Tecnologias**: Python, TensorFlow, Streamlit, Plotly
- **Última Atualização**: Dezembro 2024

⚠️ **Importante**: Este projeto é exclusivamente para fins educacionais e de pesquisa acadêmica.
""")
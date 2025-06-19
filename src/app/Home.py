import streamlit as st
import os
import sys

# Adicionar o diretório raiz do projeto ao sys.path
# Isso garante que os módulos em 'src' possam ser importados
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)

st.set_page_config(
    page_title="Análise de Ações e Previsão com IA",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.meusite.com/help',
        'Report a bug': "https://www.meusite.com/bug",
        'About': "# Aplicativo de Análise de Ações e Previsão com IA\nEste é um aplicativo para visualizar dados de ações, simular estratégias de trading e avaliar modelos de previsão."
    }
)

st.sidebar.success("Selecione uma análise ou ferramenta no menu acima.")

st.title("Bem-vindo ao Sistema de Análise de Ações e Previsão com IA 📈")

st.markdown(
    """
    Este aplicativo foi desenvolvido como parte do projeto da disciplina **MIT-510: Inteligência Artificial e Machine Learning**
    e tem como objetivo demonstrar a aplicação de técnicas de IA para análise e previsão no mercado financeiro.

    **Funcionalidades disponíveis no menu lateral:**

    - **01 Análise e Previsão**: Explore dados históricos, indicadores técnicos e previsões de preços para ativos selecionados.
    - **02 Estratégias de Trading**: Simule diferentes estratégias de trading com base nos dados históricos e previsões.
    - **03 Métricas do Modelo**: Avalie a performance dos modelos de previsão utilizados.

    Utilize o menu na barra lateral para navegar entre as diferentes seções do aplicativo.
    """
)

st.info("Lembre-se que este é um projeto educacional e as previsões e estratégias aqui apresentadas não constituem recomendação de investimento.")
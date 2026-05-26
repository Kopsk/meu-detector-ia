import streamlit as st
import joblib
import pandas as pd
import numpy as np
import string
import re

# 1. Configuração da página do Streamlit
st.set_page_config(
    page_title="Detector de IA - Advanced Learning", 
    page_icon="🤖", 
    layout="centered"
)

# 2. Função de Limpeza Avançada
def limpeza_avancada_texto(texto):
    if not isinstance(texto, str):
        return ""
    texto = re.sub(r'&\w+;', ' ', texto)
    texto = re.sub(r'\\[xX][0-9a-fA-F]{2}', ' ', texto)
    texto = re.sub(r'https?://\s*\S+|www\.\S+', ' ', texto)
    texto = re.sub(r'[^a-zA-Z0-9áàâãéèêíïóôõöúçñÁÀÂÃÉÈÊÍÏÓÔÕÖÚÇÑ\s\.,!\?\-\(\)\"\']', ' ', texto)
    texto = texto.replace('\n', ' ').replace('\r', ' ')
    texto = re.sub(r'\s+', ' ', texto)
    return texto.strip()

# 3. Função para extrair as variáveis estilométricas numéricas
def calcular_features_texto(texto):
    words = texto.split()
    word_count = len(words)
    sentences = [s for s in texto.split('.') if s.strip()]
    
    avg_word_length = np.mean([len(w) for w in words]) if word_count > 0 else 0
    avg_sentence_length = np.mean([len(s.split()) for s in sentences]) if len(sentences) > 0 else 0
    type_token_ratio = len(set(words)) / word_count if word_count > 0 else 0
    punct_density = len([c for c in texto if c in string.punctuation]) / word_count if word_count > 0 else 0
    question_ratio = texto.count('?') / word_count if word_count > 0 else 0
    
    return {
        'text': texto,
        'word_count': word_count,
        'avg_word_length': avg_word_length,
        'avg_sentence_length': avg_sentence_length,
        'type_token_ratio': type_token_ratio,
        'punct_density': punct_density,
        'question_ratio': question_ratio
    }

# ==========================================================
# INTERFACE GRÁFICA DA PLATAFORMA (Sempre renderiza na tela)
# ==========================================================
st.title("🤖 Detector de Texto Gerado por IA")
st.write("Cole o seu texto abaixo para analisar a probabilidade de ter sido escrito por um Humano ou por Inteligência Artificial.")

# 4. Carregar o modelo treinado usando joblib
@st.cache_resource
def carregar_pipeline():
    return joblib.load('detector_ia_pipeline.pkl')

modelo_carregado = True
try:
    modelo_pipeline = carregar_pipeline()
except Exception as e:
    modelo_carregado = False
    st.error(f"⚠️ Erro interno ao carregar o arquivo do modelo: {e}")
    st.info("O site carregou a interface, mas as análises estão pausadas até que o arquivo .pkl seja corrigido.")

# Caixa de texto para o usuário colar a redação
texto_usuario = st.text_area("Seu Texto:", height=250)
tamanho_atual = len(texto_usuario)

# Validação do intervalo dinamicamente na tela
if tamanho_atual > 0:
    if 1500 <= tamanho_atual <= 3000:
        st.success(f"Tamanho do texto: {tamanho_atual} caracteres. Pronto para a análise!")
    else:
        st.warning(f"Atenção: O texto está com {tamanho_atual} caracteres. O modelo foi otimizado estritamente para o intervalo de 1500 a 3000 caracteres.")

# Botão de Ação
if st.button("Analisar Texto"):
    if not modelo_carregado:
        st.error("Não é possível analisar o texto porque o modelo não foi carregado corretamente.")
    elif texto_usuario.strip() == "":
        st.error("Por favor, digite ou cole algum texto antes de analisar.")
    else:
        with st.spinner("Analisando padrões estilométricos e estrutura de caracteres..."):
            # 1. Executa a limpeza avançada no texto digitado
            texto_tratado = limpeza_avancada_texto(texto_usuario)
            
            # 2. Calcula as variáveis numéricas do texto limpo
            dict_features = calcular_features_texto(texto_tratado)
            
            # 3. Cria o DataFrame com a estrutura exata exigida pelo ColumnTransformer
            input_df = pd.DataFrame([dict_features])
            
            # 4. Faz a previsão de probabilidade
            probabilidades = modelo_pipeline.predict_proba(input_df)[0]
            prob_humano = probabilidades[0] * 100
            prob_ia = probabilidades[1] * 100
            
            # 5. Exibe os resultados visuais na tela
            st.markdown("---")
            st.subheader("📊 Resultado da Análise")
            
            col1, col2 = st.columns(2)
            col1.metric(label="✍️ Probabilidade Humana", value=f"{prob_humano:.2f}%")
            col2.metric(label="🤖 Probabilidade de IA", value=f"{prob_ia:.2f}%")
            
            st.progress(int(prob_ia))
            
            if prob_ia > 65:
                st.error("⚠️ **Alta probabilidade do texto ter sido gerado por uma Inteligência Artificial.**")
            elif prob_ia > 35:
                st.warning("⚡ **O texto possui características mistas ou o resultado é inconclusivo.**")
            else:
                st.success("✨ **Alta probabilidade do texto ter sido escrito legitimamente por um Humano.**")

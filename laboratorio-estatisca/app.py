import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import os
import minhastats as ms

st.set_page_config(page_title="Laboratório Estatístico", layout="wide")
st.title("🧮 Laboratório Estatístico Interativo")

if not os.path.exists("dados.csv"):
    st.error("⚠️ O arquivo 'dados.csv' não foi encontrado na pasta do projeto!")
    st.info("Execute no seu terminal: `python gerar_dados.py` para criar a base de dados.")
    st.stop()

@st.cache_data
def carregar_dados():
    return pd.read_csv("dados.csv").dropna()

df = carregar_dados()
cols_num = df.select_dtypes(include=[np.number]).columns.tolist()
cols_cat = df.select_dtypes(exclude=[np.number]).columns.tolist()

# ... (o restante das abas segue igual)

aba1, aba2, aba3, aba4 = st.tabs([
    "📊 Estatística Descritiva", 
    "🎲 Simulações Monte Carlo", 
    "📐 Distribuições Teóricas", 
    "📈 Regressão Linear"
])

# MÓDULO 2: DESCRITIVA
with aba1:
    st.subheader("Análise Descritiva Interativa")
    tipo_var = st.radio("Tipo de Variável:", ["Numérica", "Categórica"], horizontal=True)
    
    if tipo_var == "Numérica":
        col_sel = st.selectbox("Escolha a variável numérica:", cols_num)
        valores = df[col_sel].tolist()
        
        # Medidas calculadas "na unha"
        med = ms.media(valores)
        mdn = ms.mediana(valores)
        dp = ms.desvio_padrao(valores, "amostral")
        q1, q2, q3 = ms.quartis(valores)
        iqr = q3 - q1
        lim_inf = q1 - 1.5 * iqr
        lim_sup = q3 + 1.5 * iqr
        outliers = [x for x in valores if x < lim_inf or x > lim_sup]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Média", f"{med:.2f}")
        c2.metric("Mediana", f"{mdn:.2f}")
        c3.metric("Desvio Padrão", f"{dp:.2f}")
        c4.metric("Outliers (IQR)", f"{len(outliers)} reg.")

        # Diagnóstico textual de assimetria
        if abs(med - mdn) < (0.05 * dp):
            assimetria = "Aproximadamente Simétrica"
        elif med > mdn:
            assimetria = "Assimetria Positiva (Cauda à direita)"
        else:
            assimetria = "Assimetria Negativa (Cauda à esquerda)"
        st.info(f"**Interpretação:** A distribuição é **{assimetria}**.")

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].hist(valores, bins=25, color="skyblue", edgecolor="black")
        ax[0].set_title("Histograma")
        ax[1].boxplot(valores, vert=False)
        ax[1].set_title("Boxplot")
        st.pyplot(fig)

    else:
        col_sel = st.selectbox("Escolha a variável categórica:", cols_cat)
        contagem = df[col_sel].value_counts().head(10)
        st.dataframe(contagem.rename("Frequência"))
        fig, ax = plt.subplots(figsize=(8, 3))
        contagem.plot(kind="bar", ax=ax, color="coral")
        ax.set_ylabel("Contagem")
        st.pyplot(fig)

# MÓDULO 3: SIMULAÇÃO (LGN E TCL)
with aba2:
    st.subheader("Probabilidade e Monte Carlo")
    sim_tipo = st.selectbox("Selecione a Simulação:", ["Lei dos Grandes Números (Moeda)", "Teorema Central do Limite"])
    
    if sim_tipo == "Lei dos Grandes Números (Moeda)":
        n_lancamentos = st.slider("Número de lançamentos:", 100, 10000, 1000, step=100)
        np.random.seed(42)
        sorteios = np.random.choice([0, 1], size=n_lancamentos) # 1 = Cara
        medias_acumuladas = np.cumsum(sorteios) / np.arange(1, n_lancamentos + 1)
        
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(medias_acumuladas, label="Frequência Relativa de Caras")
        ax.axhline(0.5, color='red', linestyle='--', label="Probabilidade Teórica (0.5)")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Lançamentos")
        ax.legend()
        st.pyplot(fig)
        
    else:
        var_tcl = st.selectbox("Variável base para o TCL:", cols_num)
        tam_amostra = st.slider("Tamanho de cada amostra (n):", 5, 100, 30)
        n_amostragens = st.slider("Número de repetições:", 100, 3000, 1000, step=100)
        
        populacao = df[var_tcl].values
        np.random.seed(42)
        medias_amostrais = [ms.media(np.random.choice(populacao, size=tam_amostra, replace=True)) for _ in range(n_amostragens)]
        
        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.hist(medias_amostrais, bins=30, density=True, color="lightgreen", edgecolor="black")
        ax.set_title(f"Médias Amostrais (n={tam_amostra}) aproximando-se da Normal")
        st.pyplot(fig)

# MÓDULO 4: DISTRIBUIÇÕES TEÓRICAS
with aba3:
    st.subheader("Ajuste de Distribuições Teóricas")
    var_dist = st.selectbox("Variável para ajuste:", cols_num, key="dist")
    tipo_dist = st.selectbox("Distribuição Teórica:", ["Normal", "Exponencial"])
    
    amostra = df[var_dist].dropna().values
    m_calc = ms.media(amostra)
    s_calc = ms.desvio_padrao(amostra, "amostral")
    
    fig, ax = plt.subplots(figsize=(8, 4))
    count, bins, _ = ax.hist(amostra, bins=30, density=True, alpha=0.6, color="gray", edgecolor="black")
    x_eixo = np.linspace(min(amostra), max(amostra), 200)
    
    if tipo_dist == "Normal":
        pdf = stats.norm.pdf(x_eixo, loc=m_calc, scale=s_calc)
        ax.plot(x_eixo, pdf, 'r-', lw=2, label=f"Normal (μ={m_calc:.1f}, σ={s_calc:.1f})")
    else:
        taxa = 1.0 / m_calc if m_calc > 0 else 1.0
        pdf = stats.expon.pdf(x_eixo, scale=1.0/taxa)
        ax.plot(x_eixo, pdf, 'b-', lw=2, label=f"Exponencial (λ={taxa:.4f})")
        
    ax.legend()
    st.pyplot(fig)

# MÓDULO 5: CORRELAÇÃO E REGRESSÃO
with aba4:
    st.subheader("Regressão Linear Simples")
    var_x = st.selectbox("Variável Independente (X):", cols_num, index=0)
    var_y = st.selectbox("Variável Dependente (Y):", cols_num, index=min(1, len(cols_num)-1))
    
    x_vals = df[var_x].tolist()
    y_vals = df[var_y].tolist()
    
    r = ms.correlacao_pearson(x_vals, y_vals)
    b0, b1, r2 = ms.regressao_linear_simples(x_vals, y_vals)
    
    st.write(f"**Coeficiente de Pearson (r):** `{r:.4f}` | **R²:** `{r2:.4f}`")
    st.write(f"**Equação da Reta:** `Ŷ = {b0:.2f} + ({b1:.4f}) * X`")
    st.warning("⚠️ **Lembre-se:** Correlação quantifica associação linear, não implica relação de causa e efeito!")
    
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(x_vals, y_vals, alpha=0.3, color="blue", label="Dados")
    x_linha = np.linspace(min(x_vals), max(x_vals), 100)
    y_linha = [b0 + b1 * xi for xi in x_linha]
    ax.plot(x_linha, y_linha, color="red", label="Reta Ajustada")
    ax.set_xlabel(var_x)
    ax.set_ylabel(var_y)
    ax.legend()
    st.pyplot(fig)
    
    st.markdown("**Predição Interativa:**")
    val_entrada = st.number_input(f"Digite um valor para {var_x}:", value=float(ms.media(x_vals)))
    pred_y = b0 + b1 * val_entrada
    st.success(f"Valor estimado para {var_y} (Ŷ): **{pred_y:.2f}**")
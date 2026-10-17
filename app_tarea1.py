import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter

nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

st.set_page_config(page_title="Analizador de textos", layout="centered")
st.markdown("<style>.block-container {max-width: 1100px;}</style>", unsafe_allow_html=True)

st.markdown("<h1 style='text-align: center; font-weight: 400;'>Analizador de textos</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; font-weight: 700; margin-top: -10px;'>Goodreads</h3>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Sube un CSV con descripciones de libros, ajusta los parámetros y presiona <b>Ejecutar</b> para ver estadísticas.</p>", unsafe_allow_html=True)

archivo = st.file_uploader("Subir archivo CSV", type="csv")

# primero se carga el csv, después de habilita el botón ejecutar
if archivo is None:
    _, centro, _ = st.columns([2, 1, 2])
    with centro:
        st.button("Ejecutar", disabled=True, type="primary", use_container_width=True)
else:
    df = pd.read_csv(archivo)

    with st.container(border=True):
        col1, col2 = st.columns([3, 2])

        with col1:
            st.write("CSV cargado")
            st.dataframe(df, height=250)

        with col2:
            st.write("**Configura tus parámetros**")
            columna = st.selectbox("Columna de texto", df.columns, index=list(df.columns).index("texto") if "texto" in df.columns else 0)
            idioma = st.selectbox("Idioma stop-words", ["english", "spanish"])
            n_palabras = st.number_input("N° de palabras a graficar", 5, 50, 20)

    _, centro, _ = st.columns([2, 1, 2])
    with centro:
        ejecutar = st.button("Ejecutar", type="primary", use_container_width=True)

    if ejecutar:
        stop = set(stopwords.words(idioma))
        docs = df[columna].dropna().astype(str).tolist()

        clean_docs = []
        for doc in docs:
            limpios = []
            for token in word_tokenize(doc):
                tok = token.lower().strip()
                if tok.isalpha() and tok not in stop:
                    limpios.append(tok)
            clean_docs.append(limpios)

        todos_tokens = sum(clean_docs, [])
        bolsa = Counter(todos_tokens)
        lexico = sorted(bolsa.keys())

        st.markdown("<h2 style='text-align: center; font-weight: 400;'>Estos son tus resultados</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; font-weight: 700;'>{archivo.name}</p>", unsafe_allow_html=True)

        col3, col4 = st.columns(2)

        with col3:
            st.markdown("<h4 style='text-align: center; font-weight: 400;'>Estadísticas</h4>", unsafe_allow_html=True)
            st.markdown(f"Tamaño del léxico: <span style='background-color: #C4FF9E; border-radius: 12px; padding: 4px 18px; font-weight: 700;'>{len(lexico)}</span>", unsafe_allow_html=True)
            st.write("Tokens por documento:")
            nombres = df.loc[df[columna].notna(), "titulo"].tolist() if "titulo" in df.columns else list(range(len(clean_docs)))
            st.dataframe(pd.DataFrame({"documento": nombres,
                                       "tokens": [len(d) for d in clean_docs]}), hide_index=True)
            st.write("Primeras 15 palabras del léxico:")
            st.write(", ".join(lexico[:15]))

            m1, m2, m3 = st.columns(3)
            m1.metric("Documentos", len(clean_docs))
            m2.metric("Tokens totales", len(todos_tokens))
            m3.metric("Promedio por documento", round(len(todos_tokens) / len(clean_docs), 1))

            if "topico" in df.columns:
                st.write("Tokens por tópico:")
                por_topico = pd.DataFrame({"topico": df.loc[df[columna].notna(), "topico"].tolist(),
                                           "tokens": [len(d) for d in clean_docs]})
                st.dataframe(por_topico.groupby("topico")["tokens"].agg(["count", "sum", "mean"]).round(1)
                             .rename(columns={"count": "documentos", "sum": "tokens", "mean": "promedio"}))

        with col4:
            st.markdown("<h4 style='text-align: center; font-weight: 400;'>Frecuencia</h4>", unsafe_allow_html=True)
            principales = bolsa.most_common(n_palabras)
            palabras = [p for p, f in principales]
            conteos = [f for p, f in principales]

            colores = plt.cm.winter_r([0.25 + 0.5 * i / max(len(palabras) - 1, 1) for i in range(len(palabras))])
            colores = [(0.6 + 0.4 * c[0], 0.9 + 0.1 * c[1], 0.55 + 0.45 * c[2], 1) for c in colores]

            fig, ax = plt.subplots(figsize=(5, 6))
            barras = ax.barh(palabras, conteos, color=colores)
            ax.bar_label(barras, padding=3, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel("Número de apariciones")
            for borde in ["top", "right"]:
                ax.spines[borde].set_visible(False)
            st.pyplot(fig)

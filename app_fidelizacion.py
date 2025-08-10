import streamlit as st
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import MinMaxScaler
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="Demo Telco: Fidelización (NPS y Sentimiento)", layout="wide")
st.title(" Demo IA Telco — Fidelización, Satisfacción y NPS con Análisis de Texto")

st.markdown("""
Este dashboard muestra cómo la **Inteligencia Artificial (NLP y Deep Learning)** puede revolucionar la gestión de fidelización, permitiendo comprender la voz del cliente, analizar el NPS, predecir riesgo de churn y extraer automáticamente temas de interés a partir de comentarios abiertos.
""")

# ---------- 1. Simulación de Datos con Historial NPS y Sentimiento ----------
with st.expander("1️ ¿Cómo se crean los datos? (Simulación realista)"):
    st.info("""
    Se simulan registros de clientes con:
    - Comentarios abiertos
    - NPS histórico (últimos 6 meses)
    - Sentimiento histórico (últimos 6 meses)
    - Historial de compras
    - Número de quejas y flag de churn
    """)

# Palabras positivas y negativas para simular comentarios
positivas = [
    "excelente", "rápido", "amable", "calidad", "recomiendo", "satisfecho", "fácil", "eficiente", "cumplen", "solución"
]
negativas = [
    "lento", "malo", "problema", "no resuelven", "espera", "costoso", "nunca", "falla", "desilusión", "pésimo"
]
neutras = [
    "llamada", "servicio", "producto", "información", "consulté", "proceso", "respuesta", "asistencia"
]

def simular_comentario(sentimiento):
    palabras = []
    if sentimiento == "positivo":
        palabras = random.choices(positivas, k=3) + random.choices(neutras, k=2)
    elif sentimiento == "negativo":
        palabras = random.choices(negativas, k=3) + random.choices(neutras, k=2)
    else:
        palabras = random.choices(neutras, k=5)
    random.shuffle(palabras)
    return " ".join(palabras).capitalize() + "."

N = 600
clientes_id = [f"C{str(i).zfill(4)}" for i in range(N)]
meses = ['2024-01', '2024-02', '2024-03', '2024-04', '2024-05', '2024-06']
sent_hist = []
nps_hist = []

for c in range(N):
    sent_cliente = np.random.choice(["positivo", "negativo", "neutral"], len(meses), p=[0.5, 0.3, 0.2])
    nps_cliente = []
    for s in sent_cliente:
        if s == "positivo":
            nps_cliente.append(np.random.randint(9, 11))
        elif s == "negativo":
            nps_cliente.append(np.random.randint(0, 7))
        else:
            nps_cliente.append(np.random.randint(7, 9))
    sent_hist.append(sent_cliente)
    nps_hist.append(nps_cliente)

# Último mes
sentimientos = [x[-1] for x in sent_hist]
nps = [x[-1] for x in nps_hist]
historial_compras = np.random.poisson(3, N)
quejas = np.random.binomial(4, 0.18, N)
data = pd.DataFrame({
    "ClienteID": clientes_id,
    "Comentario": [simular_comentario(s) for s in sentimientos],
    "Sentimiento_real": sentimientos,
    "NPS": nps,
    "Historial_compras": historial_compras,
    "Quejas": quejas
})
# Churn simulado
data["Churn_predicho"] = ((data["NPS"] < 7) | (data["Quejas"] > 2)).astype(int)

# Para tendencia: DataFrame largo con meses
data_trend = []
for i, cid in enumerate(clientes_id):
    for idx, mes in enumerate(meses):
        data_trend.append({
            "ClienteID": cid,
            "Mes": mes,
            "NPS": nps_hist[i][idx],
            "Sentimiento": sent_hist[i][idx],
            "Churn_predicho": int((nps_hist[i][idx] < 7) or (quejas[i] > 2))
        })
data_trend = pd.DataFrame(data_trend)

# ---------- 2. Análisis de Sentimiento ----------
with st.expander("2️ Análisis de sentimiento con Deep Learning (simulado BERT/distilBERT)"):
    st.info("""
    Se simula la aplicación de un modelo de sentimiento tipo BERT, altamente preciso para analizar comentarios en español y detectar si el cliente está satisfecho, insatisfecho o neutral.
    """)
    resumen = data["Sentimiento_real"].value_counts(normalize=True).reindex(["positivo","negativo","neutral"], fill_value=0)
    st.markdown("**Distribución de sentimientos detectados en los comentarios:**")
    fig_sent = px.pie(values=resumen.values, names=resumen.index,
                      color=resumen.index, color_discrete_map={"positivo":"green", "negativo":"red", "neutral":"gray"},
                      title="Análisis de sentimiento por IA (simulación BERT/distilBERT)")
    st.plotly_chart(fig_sent, use_container_width=True)
    st.markdown(f"""
    - 🟢 **Positivo:** {resumen['positivo']:.0%}  
    - 🔴 **Negativo:** {resumen['negativo']:.0%}  
    - ⚪ **Neutral:** {resumen['neutral']:.0%}  
    """)

# ---------- 3. Nube de Palabras Interactiva ----------
with st.expander("3️ Nube de palabras de términos frecuentes"):
    st.info("""
    Visualiza las palabras que más mencionan los clientes en sus comentarios (mayor tamaño = más frecuencia).
    """)
    all_comments = " ".join(data["Comentario"].tolist())
    wordcloud = WordCloud(
        stopwords=STOPWORDS,
        background_color="white",
        width=800, height=350,
        colormap="Spectral", max_words=40
    ).generate(all_comments)
    fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis("off")
    st.pyplot(fig_wc)
    st.markdown("""
    **¿Cómo interpretar esto?**  
    - Las palabras grandes reflejan los temas y emociones que más importan al cliente.  
    - Puedes ajustar guiones y programas de fidelización según lo que realmente preocupa o entusiasma a tus usuarios.
    """)

# ---------- 4. Relación NPS y Churn Predicho ----------
with st.expander("4️ ¿Cómo se relacionan NPS y churn?"):
    st.info("""
    Los clientes con NPS bajo tienen mayor probabilidad de abandono (churn).  
    Aquí se visualiza cómo se relaciona la satisfacción declarada con el riesgo de perder al cliente.
    """)
    data['NPS_bin'] = pd.cut(data['NPS'], bins=[-1,6,8,10], labels=["Bajo (0-6)","Medio (7-8)","Alto (9-10)"])
    rel = data.groupby("NPS_bin")["Churn_predicho"].mean().reset_index()
    fig_nps_churn = px.bar(rel, x="NPS_bin", y="Churn_predicho", labels={"Churn_predicho":"% Churn predicho"},
                           title="Relación entre NPS y churn predicho")
    fig_nps_churn.update_traces(marker_color=["red","orange","green"])
    st.plotly_chart(fig_nps_churn, use_container_width=True)
    st.markdown("""
    - NPS **alto** = muy bajo riesgo de churn  
    - NPS **bajo** = riesgo alto, urge acción de fidelización  
    """)

# ---------- 5. Tendencia histórica de NPS y Sentimiento ----------
with st.expander("5️ Tendencia histórica animada de NPS y sentimiento"):
    st.info("""
    Analiza la evolución mensual del NPS y del sentimiento global, ideal para mostrar impacto de acciones de fidelización.
    """)
    # NPS promedio mensual
    nps_trend = data_trend.groupby("Mes")["NPS"].mean().reset_index()
    fig_nps_trend = px.line(nps_trend, x="Mes", y="NPS", markers=True, title="Tendencia histórica del NPS promedio")
    st.plotly_chart(fig_nps_trend, use_container_width=True)
    # Sentimiento mensual
    sent_trend = data_trend.groupby(["Mes","Sentimiento"]).size().reset_index(name="count")
    fig_sent_trend = px.bar(sent_trend, x="Mes", y="count", color="Sentimiento", barmode="group",
                            title="Distribución de sentimientos a lo largo del tiempo")
    st.plotly_chart(fig_sent_trend, use_container_width=True)

# ---------- 6. BONUS: Extracción de Temas (Topic Modeling Simulado) ----------
with st.expander("6️ ¿De qué hablan los clientes? (Extracción automática de temas)"):
    st.info("""
    Simulación de topic modeling: la IA agrupa los comentarios en temas frecuentes usando embeddings y clustering.
    """)
    temas = [
        "Atención al cliente", "Velocidad del servicio", "Precios y promociones",
        "Resolución de problemas", "Calidad técnica", "Facilidad de uso"
    ]
    data['Tema_comentario'] = np.random.choice(temas, N)
    tema_counts = data['Tema_comentario'].value_counts().reset_index()
    tema_counts.columns = ['Tema', 'Count']  # Corrección clave
    fig_temas = px.bar(tema_counts, x='Tema', y='Count', color='Tema',
                       title="Temas más frecuentes en los comentarios", labels={"Tema":"Tema", "Count":"N° comentarios"})
    st.plotly_chart(fig_temas, use_container_width=True)
    st.markdown("""
    **¿Para qué sirve?**  
    - Permite enfocar los esfuerzos de mejora en los temas que más afectan la percepción del cliente.
    - Los modelos reales pueden detectar automáticamente nuevos temas emergentes.
    """)

# ---------- 7. Clustering visual de clientes ----------
with st.expander("7️ Segmentación y clustering visual de clientes"):
    st.info("""
    La IA puede segmentar clientes según su perfil de satisfacción y riesgo, facilitando estrategias de retención personalizadas.
    """)
    scaler = MinMaxScaler()
    perfil_df = pd.DataFrame({
        "NPS": data["NPS"],
        "Historial_compras": data["Historial_compras"],
        "Quejas": data["Quejas"],
        "Churn_predicho": data["Churn_predicho"]
    })
    perfil_df_scaled = pd.DataFrame(scaler.fit_transform(perfil_df), columns=perfil_df.columns)
    perfil_df_scaled["Cluster"] = pd.cut(perfil_df_scaled["NPS"] + (1-perfil_df_scaled["Churn_predicho"]), bins=3, labels=["Leales", "Neutrales", "En riesgo"])
    fig_clust = px.scatter_3d(
        perfil_df_scaled, x="NPS", y="Historial_compras", z="Quejas", color="Cluster",
        title="Clustering visual de clientes por NPS, compras y quejas"
    )
    st.plotly_chart(fig_clust, use_container_width=True)
    st.markdown("""
    **¿Qué puedes hacer con esto?**  
    - Identificar visualmente grupos de clientes para campañas segmentadas.
    - Analizar la composición de los clientes en riesgo vs. los leales.
    """)

# ---------- 8. Detalle Individual (Exploración Interactiva) ----------
with st.expander("8️ Explora comentarios individuales y su análisis IA"):
    st.info("Puedes seleccionar un cliente al azar y ver el análisis detallado que hace la IA.")
    idx = st.slider("Selecciona el cliente (índice)", 0, N-1, 0)
    fila = data.iloc[idx]
    st.markdown(f"""
    - **Comentario:** _"{fila['Comentario']}"_  
    - **Sentimiento IA:** {fila['Sentimiento_real'].capitalize()}  
    - **NPS:** {fila['NPS']}  
    - **Quejas:** {fila['Quejas']}  
    - **Tema detectado:** {fila['Tema_comentario']}  
    - **Churn predicho:** {"Sí" if fila['Churn_predicho'] else "No"}
    """)

# ---------- 9. Exporta los datos del dashboard ----------
with st.expander("9️ Descarga los datos simulados del dashboard"):
    st.info("Descarga todos los registros simulados para análisis adicional o benchmarking.")
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Descargar datos (CSV)",
        data=csv,
        file_name='datos_fidelizacion_telco.csv',
        mime='text/csv'
    )

st.success("¡Listo! Este demo ultra-completo muestra todo el poder de la IA aplicada a fidelización, NPS y análisis de texto en Telco.")

##streamlit run app_fidelizacion.py
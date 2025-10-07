# app.py: Voz del Cliente 360 Plus para Droguerías (Código Final y Corregido)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import date, timedelta

# --- 0. Configuración Inicial ---
st.set_page_config(
    page_title="Voz del Cliente 360 Plus: Inteligencia para Droguerías",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Voz del Cliente 360 Plus: Servicio y Competencia")
st.subheader("La herramienta que transforma la voz del cliente en inteligencia de mercado y servicio para Droguerías.")

# --- 1. Generación de Datos Ficticios (Omnicanal) ---
@st.cache_data
def load_data():
    np.random.seed(42)  # Para reproducibilidad
    n_samples = 200
    
    # Lista base de 6 snippets para repetir (contiene ejemplos de todos los focos)
    snippet_base = [
        "El servicio fue **excelente**, los voy a **recomendar**.",
        "En **Farmacia A** el precio de esa crema es **más barato**, ¿por qué?",
        "La atención fue **pésima**, **no vuelvo a comprar** aquí.",
        "Estoy **confundido** con la receta, el chatbot no me entendió.",
        "Necesito un **Domicilio Urgente** en la noche, ¿lo ofrecen?",
        "¿Manejan la **Marca X** de vitaminas? La busqué y no la tienen."
    ]

    # Calcular repeticiones y recortar la lista (SOLUCIÓN al TypeError anterior)
    num_repeticiones = n_samples // len(snippet_base) + 1
    transcripciones = (snippet_base * num_repeticiones)[:n_samples]

    data = {
        'ID_Interaccion': range(1001, 1001 + n_samples),
        'Fecha': pd.to_datetime(np.random.choice(pd.date_range(date.today() - timedelta(days=30), date.today()), n_samples)),
        'Canal': np.random.choice(['Llamada', 'Chatbot', 'Red Social (DM)'], n_samples, p=[0.45, 0.35, 0.20]),
        'Emocion_Cliente': np.random.choice(['Frustración Extrema', 'Confusión', 'Neutral', 'Satisfacción', 'Agradecimiento'], n_samples, p=[0.10, 0.15, 0.45, 0.20, 0.10]),
        'Tipo_Opinion': np.random.choice(['Detractor', 'Promotor', 'Neutral'], n_samples, p=[0.20, 0.45, 0.35]), # Opinión sobre el servicio (iNPS)
        'Mencion_Competencia': np.random.choice([True, False], n_samples, p=[0.30, 0.70]), 
        'Competidor_Nombrado': np.random.choice(['Farmacia A', 'Droguería B', 'Ninguno'], n_samples, p=[0.15, 0.15, 0.70]),
        'Necesidad_No_Satisfecha': np.random.choice(['Marca X', 'Domicilio Urgente', 'Asesoría 24/7', 'Ninguna'], n_samples, p=[0.10, 0.10, 0.05, 0.75]),
        'Transcripcion_Snippet': transcripciones # Usamos la lista de transcripciones recortada
    }
    df = pd.DataFrame(data).head(n_samples)
    df.loc[df['Mencion_Competencia'] == False, 'Competidor_Nombrado'] = 'Ninguno'
    return df

df = load_data()

# --- 2. Filtros Laterales ---
st.sidebar.header("Filtros de Tiempo y Canal")
canal_seleccionado = st.sidebar.multiselect("Canal de Contacto", options=df['Canal'].unique(), default=df['Canal'].unique())

df_filtrado = df[df['Canal'].isin(canal_seleccionado)]

# --- 3. KPI Principales: Opinión, Competencia, Sentimiento y Demanda ---
st.header("1.Indicadores Clave de Servicio y Mercado")
col1, col2, col3, col4 = st.columns(4)

# KPI 1: Opinión General (iNPS)
promotores = df_filtrado[df_filtrado['Tipo_Opinion'] == 'Promotor'].shape[0]
detractores = df_filtrado[df_filtrado['Tipo_Opinion'] == 'Detractor'].shape[0]
iNPS_calc = 0
if df_filtrado.shape[0] > 0:
    iNPS_calc = ((promotores - detractores) / df_filtrado.shape[0]) * 100
col1.metric(
    label="iNPS (Net Promoter Score Inferido)",
    value=f"{iNPS_calc:.0f}",
    delta=f"Promotores: {promotores} | Detractores: {detractores}"
)

# KPI 2: Alertas de Competencia
menciones_comp = df_filtrado[df_filtrado['Mencion_Competencia'] == True].shape[0]
col2.metric(
    label=" Alertas de Mención de Competencia",
    value=f"{menciones_comp} Interacciones",
    delta="Inteligencia de Mercado en Tiempo Real"
)

# KPI 3: Sentimientos (Fricción)
friccion = df_filtrado[df_filtrado['Emocion_Cliente'].isin(['Frustración Extrema', 'Confusión'])].shape[0]
col3.metric(
    label=" Alta Fricción/Confusión",
    value=f"{friccion} Clientes",
    delta="Focus en la Experiencia (CX)"
)

# KPI 4: Necesidades No Satisfechas
necesidad_no_sat = df_filtrado[df_filtrado['Necesidad_No_Satisfecha'] != 'Ninguna'].shape[0]
col4.metric(
    label=" Demanda NO Satisfecha",
    value=f"{necesidad_no_sat} Solicitudes",
    delta="Oportunidad de Inventario/Servicio"
)

# ---
## 2.  Mapeo Profundo: Emoción, Precio y Servicio

col_left, col_right = st.columns(2)

with col_left:
    st.subheader(" Alertas de Competencia: ¿Quién y Por Qué?")
    menciones_comp_count = df_filtrado[df_filtrado['Competidor_Nombrado'] != 'Ninguno']['Competidor_Nombrado'].value_counts().reset_index()
    menciones_comp_count.columns = ['Competidor', 'Volumen de Alertas']
    fig_comp = px.bar(menciones_comp_count, x='Competidor', y='Volumen de Alertas',
                       title='Volumen de Alertas por Competidor Nombrado',
                       color='Competidor', color_discrete_map={'Farmacia A': 'orange', 'Droguería B': 'red', 'Ninguno': 'lightgray'})
    st.plotly_chart(fig_comp, use_container_width=True)

    st.subheader(" Demanda para Innovación")
    necesidades = df_filtrado[df_filtrado['Necesidad_No_Satisfecha'] != 'Ninguna']['Necesidad_No_Satisfecha'].value_counts().reset_index()
    necesidades.columns = ['Necesidad', 'Volumen']
    st.dataframe(necesidades, use_container_width=True, hide_index=True)


with col_right:
    st.subheader(" Fricción: Emociones y Canales")
    fig_emocion = px.histogram(df_filtrado, x='Emocion_Cliente', color='Canal', 
                               title='Distribución Emocional por Canal de Contacto',
                               color_discrete_map={'Llamada': '#FF5733', 'Chatbot': '#33FF57', 'Red Social (DM)': '#3357FF'})
    st.plotly_chart(fig_emocion, use_container_width=True)
    
    st.subheader(" Distribución de la Opinión del Servicio (iNPS)")
    fig_opinion = px.pie(df_filtrado, names='Tipo_Opinion', title='Clasificación de la Opinión del Servicio',
                         color_discrete_map={'Promotor': 'green', 'Neutral': 'blue', 'Detractor': 'red'})
    st.plotly_chart(fig_opinion, use_container_width=True)

# ---
## 3.  Evidencia Directa: Alertas y Detractores en la Voz

st.header("3. Evidencia Directa: Alertas y Detractores en la Voz")
st.markdown("Ejemplos de interacciones filtradas por **'Mención de Competencia'** o **'Detractor'**.")

df_muestras = df_filtrado[(df_filtrado['Mencion_Competencia'] == True) | (df_filtrado['Tipo_Opinion'] == 'Detractor')].head(5)

for index, row in df_muestras.iterrows():
    # Detecta el tipo de alerta más relevante para el encabezado
    alerta_tipo = ""
    if row['Competidor_Nombrado'] != 'Ninguno':
        alerta_tipo = f" ALERTA: Menciona a **{row['Competidor_Nombrado']}**"
    elif row['Tipo_Opinion'] == 'Detractor':
        alerta_tipo = " DETRACTOR: Servicio **PÉSIMO**"
    
    st.info(f"**ID: {row['ID_Interaccion']}** | **Canal:** {row['Canal']} | {alerta_tipo}")
    
    # Resaltar palabras clave
    snippet = row['Transcripcion_Snippet']
    snippet = snippet.replace("precio", "**PRECIO**").replace("más barato", "**MÁS BARATO**").replace("Farmacia A", "**FARMACIA A**").replace("recomendar", "**RECOMENDAR**").replace("pésimo servicio", "**PÉSIMO SERVICIO**").replace("Domicilio Urgente", "**DOMICILIO URGENTE**").replace("Marca X", "**MARCA X**")
    
    st.markdown(f"> **{snippet}**")
    
    # Muestra el placeholder de análisis de audio/texto
    if row['Canal'] == 'Llamada':
        # Reemplazo del widget st.audio()
        st.caption(" **ANÁLISIS DE VOZ (Llamada):** Tono detectado: Frustración Alta.") 
    elif row['Canal'] != 'Llamada':
        st.caption("*(Análisis de sentimiento por texto en canales digitales)*")
        
    st.write("---")
    
st.markdown("""
> **Mensaje Final del Demo:** Con **Voz del Cliente 360 Plus**, usted obtiene **inteligencia de mercado accionable** (alertas de competencia) y puede medir y actuar sobre la **lealtad y el servicio (iNPS)** en todos los puntos de contacto.
""")
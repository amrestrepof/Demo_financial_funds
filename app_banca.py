import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Dashboard Estratégico de Clientes", layout="wide", page_icon="🏦")

# --- FUNCIONES DE MODELADO (CON CACHE) ---
@st.cache_data
def cargar_y_preparar_datos(filepath):
    df = pd.read_csv(filepath)
    return df

@st.cache_resource
def entrenar_modelo_churn(df):
    features = ['antiguedad_meses', 'num_productos', 'saldo_cuenta', 'score_crediticio', 'quejas_ultimo_trimestre', 'usa_app_movil', 'edad']
    X = df[features]
    y = df['churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(random_state=42, class_weight='balanced')
    model.fit(X_train_scaled, y_train)
    
    # Evaluar modelo
    preds = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, preds)
    
    # Calcular probabilidades
    churn_probs = model.predict_proba(scaler.transform(X))[:, 1]
    
    return model, scaler, features, churn_probs, accuracy

@st.cache_resource
def crear_segmentacion(df):
    features_segmentacion = ['saldo_cuenta', 'num_transacciones_mes', 'antiguedad_meses', 'edad']
    X_seg = df[features_segmentacion]
    
    scaler = StandardScaler()
    X_seg_scaled = scaler.fit_transform(X_seg)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['segmento'] = kmeans.fit_predict(X_seg_scaled)
    
    # Nombres interpretables para los segmentos
    # Esto requiere un análisis de los centroides, pero lo simularemos
    segment_map = {
        0: 'Jóvenes Digitales',
        1: 'Clientes Leales y de Alto Valor',
        2: 'Clientes en Riesgo / Bajo Uso',
        3: 'Nuevos Clientes'
    }
    df['segmento_nombre'] = df['segmento'].map(segment_map)
    return df

# --- CARGA Y ENTRENAMIENTO ---
df = cargar_y_preparar_datos('clientes_banco.csv')
model_churn, scaler_churn, features_churn, churn_probs, accuracy = entrenar_modelo_churn(df)
df['probabilidad_churn'] = churn_probs
df_segmentado = crear_segmentacion(df.copy())

# --- INTERFAZ DEL DASHBOARD ---
st.title("Dashboard Estratégico de Clientes Bancarios")

tab1, tab2 = st.tabs([" Prevención de Churn", " Segmentación Proactiva"])

# --- PESTAÑA 1: MODELO DE CHURN ---
with tab1:
    st.header("Análisis y Predicción de Abandono de Clientes")
    
    col1, col2, col3 = st.columns(3)
    tasa_churn_real = df['churn'].mean() * 100
    col1.metric("Tasa de Churn Histórica", f"{tasa_churn_real:.2f}%")
    col2.metric("Precisión del Modelo (Accuracy)", f"{accuracy:.2f}")
    
    # Umbral de riesgo
    riesgo_threshold = st.slider("Selecciona el umbral de probabilidad para considerar un cliente 'en riesgo':", 0.0, 1.0, 0.6)
    df_riesgo = df[df['probabilidad_churn'] >= riesgo_threshold]
    col3.metric("Clientes en Alto Riesgo", df_riesgo.shape[0])

    st.markdown("---")
    
    st.subheader("Clientes con Mayor Probabilidad de Abandono")
    st.dataframe(
        df.sort_values('probabilidad_churn', ascending=False).head(10)[['id_cliente', 'probabilidad_churn'] + features_churn],
        use_container_width=True
    )

    st.subheader("Factores Clave que Influyen en el Churn")
    importancias = pd.DataFrame(data={
        'feature': features_churn,
        'importance': model_churn.coef_[0]
    }).sort_values('importance', ascending=False)
    
    fig_importancia = px.bar(importancias, x='feature', y='importance', title='Importancia de cada factor en el modelo de Churn')
    st.plotly_chart(fig_importancia, use_container_width=True)
    st.info("""
    **Interpretación:**
    - **Barras Positivas:** Aumentan la probabilidad de churn (ej. muchas quejas, pocos productos).
    - **Barras Negativas:** Disminuyen la probabilidad de churn (ej. alta antigüedad, ser usuario de la app).
    """)


# --- PESTAÑA 2: SEGMENTACIÓN DE CLIENTES ---
with tab2:
    st.header("Estrategias Proactivas Basadas en Segmentación")

    st.sidebar.header("Filtros de Segmentación")
    segmento_seleccionado = st.sidebar.multiselect(
        "Filtrar por Segmento:",
        options=df_segmentado['segmento_nombre'].unique(),
        default=df_segmentado['segmento_nombre'].unique()
    )
    df_filtrado_seg = df_segmentado[df_segmentado['segmento_nombre'].isin(segmento_seleccionado)]

    st.subheader("Visualización de Segmentos de Clientes")
    fig_segmentos = px.scatter(
        df_filtrado_seg, 
        x='saldo_cuenta', 
        y='num_transacciones_mes', 
        color='segmento_nombre',
        title='Segmentos por Comportamiento Transaccional',
        hover_data=['id_cliente', 'edad', 'antiguedad_meses']
    )
    st.plotly_chart(fig_segmentos, use_container_width=True)

    st.subheader("Descripción y Estrategias por Segmento")
    
    for segmento in df_segmentado['segmento_nombre'].unique():
        with st.expander(f"**Análisis del Segmento: {segmento}**"):
            df_s = df_segmentado[df_segmentado['segmento_nombre'] == segmento]
            
            st.write(f"**Número de clientes:** {df_s.shape[0]}")
            st.write(f"**Tasa de churn promedio:** {df_s['churn'].mean()*100:.2f}%")
            
            # Descripción y Estrategia (texto prescriptivo)
            if segmento == 'Clientes Leales y de Alto Valor':
                st.success("**Estrategia Recomendada:** Programas de lealtad, acceso a gestores personales, ofertas de inversión exclusivas. El objetivo es retener y maximizar su valor.")
            elif segmento == 'Jóvenes Digitales':
                st.info("**Estrategia Recomendada:** Campañas de marketing en redes sociales, promoción de la app móvil, ofertas de productos de crédito iniciales y micro-inversiones.")
            elif segmento == 'Clientes en Riesgo / Bajo Uso':
                st.warning("**Estrategia Recomendada:** Campañas de reactivación, encuestas de satisfacción para entender sus puntos de dolor, ofertas para aumentar su vinculación (ej. domiciliar nómina).")
            elif segmento == 'Nuevos Clientes':
                st.info("**Estrategia Recomendada:** Proceso de onboarding guiado, tutoriales de la app, oferta de un segundo producto (ej. tarjeta de crédito) tras los primeros 3 meses.")

            st.dataframe(df_s.describe().T[['mean', 'std', 'min', 'max']], use_container_width=True)
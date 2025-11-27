import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Gestor de Estrategias de Retención", layout="wide", page_icon="🎯")

# --- 1. GENERACIÓN DE DATOS (Simulando la realidad de Seguros) ---
@st.cache_data
def generar_data_historica(n=2000):
    np.random.seed(42)
    
    # Simular categorías de Churn que Seguros YA TIENE (Input)
    categorias_churn_seguros = ['Riesgo Crítico', 'Riesgo Alto', 'Riesgo Medio', 'Riesgo Bajo']
    
    data = {
        'id_cliente': range(1000, 1000 + n),
        'segmento_seguros': np.random.choice(categorias_churn_seguros, n, p=[0.1, 0.2, 0.4, 0.3]),
        'producto': np.random.choice(['Auto', 'Hogar', 'Vida', 'Salud'], n),
        'antiguedad_meses': np.random.randint(6, 120, n),
        'valor_prima': np.random.normal(1500000, 400000, n),
        'siniestros_historicos': np.random.choice([0, 1, 2, 3], n, p=[0.6, 0.25, 0.1, 0.05]),
        'canal_preferido': np.random.choice(['App', 'Call Center', 'Oficina', 'WhatsApp'], n),
        'uso_app_frecuencia': np.random.choice(['Alta', 'Media', 'Baja', 'Nula'], n), # Nueva variable para input
        'nps_ultimo': np.random.randint(0, 11, n), # Nueva variable para input
        # Histórico: ¿Qué estrategia se usó en el pasado y funcionó?
        'estrategia_historica': np.random.choice(['Descuento 10%', 'Mejora Cobertura', 'Llamada Fidelización', 'Ninguna'], n),
        'resultado_historico': np.random.choice(['Retenido', 'Cancelado'], n)
    }
    
    df = pd.DataFrame(data)
    
    # Lógica de negocio para entrenar el modelo "Predictor de Estrategia"
    def definir_mejor_accion(row):
        if row['segmento_seguros'] in ['Riesgo Crítico'] and row['valor_prima'] > 1800000:
            return 'Descuento Agresivo (15%)'
        elif row['siniestros_historicos'] > 0:
            return 'Asistencia/Acompañamiento'
        elif row['segmento_seguros'] == 'Riesgo Medio' and row['canal_preferido'] == 'App':
            return 'Upgrade Digital (Cross-sell)'
        else:
            return 'Fidelización Soft (Email)'

    df['mejor_estrategia_real'] = df.apply(definir_mejor_accion, axis=1)
    
    # Propensión a ser retenido (Score 0-100 independientemente del Churn)
    # Calculado con NPS y Uso de App para darle realismo a la explicación de inputs
    df['propension_retencion'] = (df['nps_ultimo'] / 10 * 0.5) + (np.where(df['uso_app_frecuencia']=='Alta', 0.4, 0.1)) + np.random.normal(0, 0.1, n)
    df['propension_retencion'] = df['propension_retencion'].clip(0, 1)
    
    return df

# --- 2. MODELO PRESCRIPTIVO (Next Best Action) ---
@st.cache_resource
def entrenar_recomendador(df):
    le = LabelEncoder()
    df['producto_code'] = le.fit_transform(df['producto'])
    
    features = ['valor_prima', 'antiguedad_meses', 'siniestros_historicos', 'producto_code', 'propension_retencion']
    X = df[features]
    y = df['mejor_estrategia_real']
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    return model, features

# --- CARGA ---
df = generar_data_historica()
model_nba, features_nba = entrenar_recomendador(df)
df['Estrategia_Recomendada_IA'] = model_nba.predict(df[features_nba])

# --- DASHBOARD ---
st.title("Hub de Inteligencia de Retención")
st.markdown("""
Plataforma de Analítica Prescriptiva para la gestión del ciclo de vida del cliente.
""")

# TABS PRINCIPALES (Ahora son 4)
tab_specs, tab_accion, tab_propension, tab_historico = st.tabs([
    "Requerimientos de Datos (Input)",
    "1. Estrategia (Predicción)", 
    "2. Categorización (Propensión)", 
    "3. Históricos del Servicio"
])

# --- TAB 0: ESPECIFICACIONES DE DATOS (NUEVO) ---
with tab_specs:
    st.header("Diccionario de Datos para el Modelo")
    st.markdown("""
    Para que el modelo de **Analítica Prescriptiva** funcione y genere las recomendaciones de la Pestaña 2 y 3, 
    es necesario alimentar el sistema con las siguientes 4 dimensiones de información del cliente.
    """)

    col_spec1, col_spec2 = st.columns(2)

    with col_spec1:
        with st.expander("1. Perfil y Valor (Hard Data)", expanded=True):
            st.markdown("""
            *Define quién es el cliente y cuánto vale para la compañía.*
            | Variable | Descripción | Uso en Modelo |
            | :--- | :--- | :--- |
            | **ID_Cliente** | Identificador único | Llave de cruce |
            | **Segmento_Churn** | Input Actual de seguros | Nivel de Riesgo Base |
            | **Valor_Prima** | Monto anualizado | Priorización ($) |
            | **Antigüedad** | Meses activo | Sensibilidad al precio |
            | **Siniestralidad** | % Histórico | **Filtro de Rentabilidad** |
            """)
        
        with st.expander("2. Comportamiento (Soft Data)"):
            st.markdown("""
            *Define la vinculación emocional y digital del cliente.*
            | Variable | Descripción | Uso en Modelo |
            | :--- | :--- | :--- |
            | **NPS_Ultimo** | Score 0-10 | Cálculo de Propensión |
            | **Uso_App** | Frecuencia | Afinidad canal digital |
            | **Quejas_Abiertas** | Binario (Si/No) | Bloqueo de venta |
            | **Canal_Preferido** | App/Tel/Email | Selección de vía de contacto |
            """)

    with col_spec2:
        with st.expander("3. Histórico de Retención (Learning)", expanded=True):
            st.info("**Variable Crítica:** El modelo aprende de los éxitos y fracasos pasados.")
            st.markdown("""
            | Variable | Descripción | Uso en Modelo |
            | :--- | :--- | :--- |
            | **Accion_Pasada** | ¿Qué se le ofreció? | Entrenamiento del Motor |
            | **Respuesta** | ¿Renovó o Canceló? | Etiqueta de Éxito (Target) |
            | **Motivo_Baja** | Razón (Precio/Servicio) | Analítica Descriptiva |
            """)

        st.markdown("###  Plantilla de Ingesta")
        st.write("Descargue la estructura requerida para cargar los datos de seguros en este motor.")
        
        # Generar CSV dummy para descarga
        csv = df.head(10).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Descargar Layout de Datos (.csv)",
            data=csv,
            file_name='layout_input_retencion.csv',
            mime='text/csv',
        )

# --- TAB 1: PREDECIR MEJOR ESTRATEGIA ---
with tab_accion:
    st.header("Modelo Prescriptivo: Next Best Action")
    
    col_a1, col_a2 = st.columns([2, 1])
    with col_a1:
        st.subheader("Listado de Acciones Recomendadas")
        filtro_riesgo = st.multiselect("Filtrar por Riesgo (Input seguros)", df['segmento_seguros'].unique(), default=['Riesgo Crítico', 'Riesgo Alto'])
        df_view = df[df['segmento_seguros'].isin(filtro_riesgo)].head(100)
        
        st.dataframe(
            df_view[['id_cliente', 'segmento_seguros', 'producto', 'propension_retencion', 'Estrategia_Recomendada_IA']],
            column_config={
                "propension_retencion": st.column_config.ProgressColumn("Propensión a Retener", format="%.2f", min_value=0, max_value=1),
                "Estrategia_Recomendada_IA": st.column_config.TextColumn("Mejor Estrategia (IA)", help="Acción con mayor prob. de éxito")
            },
            use_container_width=True
        )
    with col_a2:
        st.subheader("Mix de Estrategias")
        fig_pie = px.pie(df_view, names='Estrategia_Recomendada_IA', title='Distribución de Acciones Sugeridas', hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)

# --- TAB 2: CATEGORIZACIÓN ---
with tab_propension:
    st.header("Matriz de Retenibilidad")
    
    col_p1, col_p2 = st.columns([3, 1])
    with col_p1:
        fig_scatter = px.scatter(df.sample(500), x='segmento_seguros', y='propension_retencion', 
                                 color='Estrategia_Recomendada_IA', size='valor_prima',
                                 title="Categorización: Riesgo (X) vs. Propensión (Y)",
                                 category_orders={'segmento_seguros': ['Riesgo Bajo', 'Riesgo Medio', 'Riesgo Alto', 'Riesgo Crítico']})
        st.plotly_chart(fig_scatter, use_container_width=True)
    with col_p2:
        st.info("Esta matriz cruza el **Input de Churn** (lo que ya saben) con la **Probabilidad de Éxito** calculada con las variables de Comportamiento (NPS, App).")

# --- TAB 3: HISTÓRICOS ---
with tab_historico:
    st.header("Analítica Descriptiva del Servicio")
    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.subheader("Flujo de Cancelaciones")
        fig_sun = px.sunburst(df, path=['producto', 'resultado_historico', 'segmento_seguros'])
        st.plotly_chart(fig_sun, use_container_width=True)
    with col_h2:
        st.subheader("Efectividad Histórica")
        efectividad = df.groupby('estrategia_historica')['resultado_historico'].apply(lambda x: (x=='Retenido').mean()).reset_index()
        fig_bar = px.bar(efectividad, x='estrategia_historica', y='resultado_historico', title="Tasa de Éxito Real", text_auto='.0%')
        st.plotly_chart(fig_bar, use_container_width=True)

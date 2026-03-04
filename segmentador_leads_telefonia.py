import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. CONFIGURACIÓN DE LA INTERFAZ
# ==========================================
st.set_page_config(
    page_title="Telefonia Lead Scoring System",
    page_icon="",
    layout="wide"
)

# Estilo personalizado para mejorar la estética
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. GENERACIÓN Y DOCUMENTACIÓN DE DATOS (Simulación de Negocio)
# ==========================================
@st.cache_data
def generar_datos_simulados(n=800):
    """
    Crea un dataset que simula el comportamiento real de una operadora.
    Variables:
    - Demográficas: Edad, Región, Ingresos.
    - Operativas: Consumo de datos (GB), Minutos, Reclamaciones.
    - Históricas: Antigüedad, Paquetes contratados.
    """
    np.random.seed(42)
    regiones = ['Norte', 'Sur', 'Centro', 'Este']
    canales = ['Web', 'App', 'Tienda Física', 'Telemarketing']
    
    data = {
        'ID_Lead': [f'TEL-{i:05d}' for i in range(n)],
        'Tipo_Lead': np.random.choice(['Prospecto Nuevo', 'Cliente Actual'], n, p=[0.4, 0.6]),
        'Edad': np.random.randint(18, 75, n),
        'Region': np.random.choice(regiones, n),
        'Ingresos_Mensuales': np.random.uniform(800, 6000, n),
        'Consumo_GB': np.random.uniform(2, 80, n),
        'Minutos_Voz': np.random.uniform(50, 1500, n),
        'Antiguedad_Meses': np.random.randint(0, 72, n),
        'Nivel_Satisfaccion': np.random.randint(1, 10, n),
        'Canal_Origen': np.random.choice(canales, n)
    }
    
    df = pd.DataFrame(data)
    
    # Lógica de negocio para simular 'Target' (Probabilidad real de venta)
    # Ejemplo: Si consume mucho pero tiene plan bajo (Upsell) o si es joven con ingresos (Conversión)
    score_base = (df['Ingresos_Mensuales'] / 6000) * 0.3 + (df['Consumo_GB'] / 80) * 0.4 + (df['Nivel_Satisfaccion'] / 10) * 0.3
    df['Venta_Exitosa'] = (score_base + np.random.normal(0, 0.1, n) > 0.6).astype(int)
    
    return df

df = generar_datos_simulados()

# ==========================================
# 3. MOTOR DE INTELIGENCIA (Modelo de Propensión)
# ==========================================
def entrenar_modelo_propension(data):
    # Preparamos los datos para un Random Forest simplificado
    le = LabelEncoder()
    temp_df = data.copy()
    
    # Codificar variables categóricas
    for col in ['Tipo_Lead', 'Region', 'Canal_Origen']:
        temp_df[col] = le.fit_transform(temp_df[col])
        
    X = temp_df.drop(['ID_Lead', 'Venta_Exitosa'], axis=1)
    y = temp_df['Venta_Exitosa']
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    # Generar el Score de 0 a 100%
    data['Score_Propension'] = (model.predict_proba(X)[:, 1] * 100).round(2)
    return model, data

model, df_scored = entrenar_modelo_propension(df)

# ==========================================
# 4. DISEÑO DEL DASHBOARD (Frontend)
# ==========================================

# --- Encabezado ---
st.title(" Sistema de Inteligencia de Clientes - Telecom")
st.markdown("""
    **Propósito:** Esta plataforma identifica qué leads tienen mayor probabilidad de **conversión** (clientes nuevos) 
    o **upselling** (clientes actuales comprando más), permitiendo optimizar el esfuerzo del equipo comercial.
""")
st.divider()

# --- KPIs Principales ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Leads Analizados", len(df_scored))
with col2:
    high_value = len(df_scored[df_scored['Score_Propension'] > 75])
    st.metric("Prioridad Alta (>75%)", high_value, delta=f"{high_value/len(df_scored)*100:.1f}% del total")
with col3:
    st.metric("Promedio GB Consumo", f"{df_scored['Consumo_GB'].mean():.1f} GB")
with col4:
    st.metric("Ingreso Promedio Est.", f"${df_scored['Ingresos_Mensuales'].mean():,.0f}")

# --- Pestañas de Navegación ---
tab1, tab2, tab3 = st.tabs([" Análisis de Segmentos", " Explorador de Leads", " Simulador de Score"])

# TAB 1: VISUALIZACIONES ESTRATÉGICAS
with tab1:
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Distribución de Probabilidad por Tipo de Lead")
        fig_box = px.box(df_scored, x='Tipo_Lead', y='Score_Propension', color='Tipo_Lead',
                         points="all", title="Propensión: Venta Nueva vs Upsell")
        st.plotly_chart(fig_box, use_container_width=True)
        
    with c2:
        st.subheader("Relación Consumo vs Ingresos vs Score")
        fig_scatter = px.scatter(df_scored, x="Ingresos_Mensuales", y="Consumo_GB", 
                                 color="Score_Propension", size="Nivel_Satisfaccion",
                                 color_continuous_scale='RdYlGn', title="Matriz de Valor de Lead")
        st.plotly_chart(fig_scatter, use_container_width=True)

# TAB 2: EXPLORADOR DE DATOS
with tab2:
    st.subheader("Listado de Leads Priorizados")
    
    # Filtros rápidos
    f_tipo = st.multiselect("Filtrar por Tipo:", df_scored['Tipo_Lead'].unique(), default=df_scored['Tipo_Lead'].unique())
    f_min_score = st.slider("Mínimo Score de Propensión:", 0, 100, 60)
    
    df_filtrado = df_scored[(df_scored['Tipo_Lead'].isin(f_tipo)) & (df_scored['Score_Propension'] >= f_min_score)]
    
    st.dataframe(df_filtrado.sort_values('Score_Propension', ascending=False), use_container_width=True)
    
    st.download_button("Descargar Lista para Call Center (CSV)", 
                       df_filtrado.to_csv(index=False), "leads_priorizados.csv", "text/csv")

# TAB 3: SIMULADOR PARA EL CLIENTE
with tab3:
    st.subheader("Simulador de Propensión (What-if)")
    st.write("Ingrese los datos de un prospecto para predecir su éxito comercial.")
    
    with st.expander("Abrir Formulario de Entrada"):
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            s_tipo = st.selectbox("Situación del Lead", ["Prospecto Nuevo", "Cliente Actual"])
            s_edad = st.number_input("Edad del Sujeto", 18, 90, 35)
            s_reg = st.selectbox("Región Geográfica", ['Norte', 'Sur', 'Centro', 'Este'])
        with sc2:
            s_ing = st.slider("Ingresos Mensuales Estimados", 500, 10000, 2500)
            s_gb = st.slider("Uso de Datos (GB/Mes)", 0, 200, 20)
            s_min = st.number_input("Minutos de Voz/Mes", 0, 5000, 400)
        with sc3:
            s_ant = st.number_input("Antigüedad (Meses)", 0, 240, 12)
            s_sat = st.slider("Nivel de Satisfacción (1-10)", 1, 10, 7)
            s_can = st.selectbox("Canal de Contacto", ['Web', 'App', 'Tienda Física', 'Telemarketing'])

    if st.button(" Calcular Probabilidad de Éxito"):
        # Mostrar resultado con un indicador visual (Gauge)
        # Nota: Aquí se usaría el modelo entrenado con los inputs del simulador
        # Para efectos prácticos, generamos un score basado en los inputs:
        res_score = (s_ing/10000)*20 + (s_gb/200)*40 + (s_sat*4) # Lógica simplificada simulada
        res_score = min(max(res_score, 10), 98.5)
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = res_score,
            title = {'text': "Probabilidad de Venta (%)"},
            gauge = {
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1f77b4"},
                'steps': [
                    {'range': [0, 40], 'color': "#ffcfcf"},
                    {'range': [40, 70], 'color': "#fff4cf"},
                    {'range': [70, 100], 'color': "#cfdfcf"}
                ],
                'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}
            }
        ))
        st.plotly_chart(fig_gauge)
        
        if res_score > 70:
            st.success("✅ **RECOMENDACIÓN:** Lead de alta prioridad. Ofrecer Plan Premium de inmediato.")
        elif res_score > 40:
            st.warning("⚠️ **RECOMENDACIÓN:** Lead interesado. Enviar campaña (Lead Nurturing) por WhatsApp/Email.")
        else:
            st.error("❌ **RECOMENDACIÓN:** Baja probabilidad. No priorizar en llamadas salientes.")
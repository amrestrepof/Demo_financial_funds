import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="RutaFresca - Analytics AI", layout="wide", page_icon="🚚")

# --- 1. GENERACIÓN DE DATOS ---
@st.cache_data
def generar_datos(n_clientes):
    np.random.seed(42)
    ids = [f"CTE-{i:04d}" for i in range(n_clientes)]
    nombres = [f"Tienda {np.random.choice(['El Vecino','La Esperanza','Don Pepe','La Esquina','El Paisa','Santa Maria','El Progreso'])} {i}" for i in range(n_clientes)]
    
    # Simulación de Comportamiento
    recencia = np.random.randint(1, 90, n_clientes) 
    frecuencia_mensual = np.random.randint(1, 8, n_clientes) 
    ticket_promedio = np.random.uniform(50, 400, n_clientes) * 1000 
    
    # Cálculos monetarios
    venta_mensual = ticket_promedio * frecuencia_mensual
    venta_semestral = venta_mensual * 6

    # Coordenadas geográficas (Simulación)
    lat = np.random.normal(4.65, 0.02, n_clientes)
    lon = np.random.normal(-74.08, 0.02, n_clientes)

    df = pd.DataFrame({
        'Cliente_ID': ids,
        'Nombre_Tienda': nombres,
        'Recencia_Dias': recencia,
        'Frecuencia_Mensual': frecuencia_mensual,
        'Venta_Mensual': venta_mensual,
        'Venta_Total': venta_semestral,
        'lat': lat, 'lon': lon
    })
    return df

# --- 2. CEREBRO ANALÍTICO (Reglas + Machine Learning) ---
def procesar_inteligencia(df):
    # --- A. LÓGICA DE NEGOCIO (RFM & ILC) ---
    # R_Score: Recencia (Quintiles)
    df['R_Score'] = pd.qcut(df['Recencia_Dias'], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    # F_Score: Ranking Percentil (Evita errores si hay muchos datos iguales)
    df['F_Score'] = (df['Frecuencia_Mensual'].rank(pct=True) * 5).apply(np.ceil).astype(int)
    # M_Score: Monto (Quintiles)
    df['M_Score'] = pd.qcut(df['Venta_Total'], 5, labels=[1, 2, 3, 4, 5]).astype(int)

    # Score ILC (Índice de Lealtad Comercial)
    df['ILC_Score'] = ((df['R_Score']*0.3 + df['F_Score']*0.3 + df['M_Score']*0.4) / 5) * 100
    df['ILC_Score'] = df['ILC_Score'].round(1)

    # Segmentación Automática (Reglas)
    def asignar_segmento(row):
        if row['Recencia_Dias'] > 45: return 'En Peligro'
        if row['ILC_Score'] >= 85: return 'Socio Estratégico'
        if row['ILC_Score'] >= 60: return 'Cliente Regular'
        if row['ILC_Score'] >= 40: return 'Ocasional'
        return 'Bajo Potencial'

    df['Segmento'] = df.apply(asignar_segmento, axis=1)

    # --- B. INTELIGENCIA ARTIFICIAL (CLUSTERING K-MEANS) ---
    # Normalizamos los datos para que K-Means funcione bien
    scaler = MinMaxScaler()
    features = df[['Recencia_Dias', 'Frecuencia_Mensual', 'Venta_Mensual']]
    features_scaled = scaler.fit_transform(features)
    
    # Buscamos 4 Clusters (Grupos naturales)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Cluster_ID'] = kmeans.fit_predict(features_scaled)
    
    # Asignamos nombres amigables a los clusters encontrados
    cluster_map = {0: "Estándar/Recurrentes", 1: " Cuentas AAA", 2: "Inactivos/Baja Rotación", 3: " Potenciales"}
    df['Cluster_IA'] = df['Cluster_ID'].map(cluster_map).fillna("Otros")

    # --- C. ANALÍTICA PREDICTIVA (PROBABILIDAD DE FUGA) ---
    # Simulamos un score de probabilidad basado en comportamiento reciente
    # A mayor recencia y menor frecuencia, más alta la probabilidad
    raw_prob = (df['Recencia_Dias'] * 0.7) - (df['Frecuencia_Mensual'] * 5)
    # Normalizamos entre 0 y 1 (0% a 100%)
    df['Prob_Fuga_ML'] = (raw_prob - raw_prob.min()) / (raw_prob.max() - raw_prob.min())

    return df

# --- 3. INTERFAZ GRÁFICA ---

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2821/2821828.png", width=80)
st.sidebar.title("RutaFresca")
st.sidebar.caption("Inteligencia Comercial 360°")
n_clientes = st.sidebar.slider("Clientes en Base de Datos", 100, 2000, 800)
st.sidebar.markdown("---")
st.sidebar.info("Simulando Analítica Descriptiva, Predictiva y Prescriptiva.")

# Procesamiento
raw_df = generar_datos(n_clientes)
df_final = procesar_inteligencia(raw_df)

st.title("Tablero de Control: RutaFresca Analytics")

# EXPLICACIÓN TÉCNICA
with st.expander("CÓMO FUNCIONA: Arquitectura de 3 Fases", expanded=False):
    st.markdown("""
    1.  **Fase Descriptiva (Reglas):** Calculamos el *ILC (Índice de Lealtad)* basado en Recencia, Frecuencia y Monto.
        * *Fórmula:* $$ ILC = (R \\times 30\\%) + (F \\times 30\\%) + (M \\times 40\\%) $$
    2.  **Fase Predictiva (IA):** Usamos **K-Means Clustering** para encontrar grupos ocultos y un modelo de probabilidad para predecir quién se va a ir el próximo mes.
    3.  **Fase Prescriptiva (Acción):** Simulador de ROI financiero y disparador de acciones (WhatsApp).
    """)

st.markdown("---")

# ==========================================
# FASE 1: DIAGNÓSTICO (LO QUE YA TENÍAS)
# ==========================================
st.header(" Diagnóstico de Cartera (Descriptivo)")

# KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Cartera", len(df_final))
c2.metric("Venta Mensual Promedio", f"${df_final['Venta_Mensual'].mean():,.0f}")
c3.metric("En Peligro (Regla Negocio)", len(df_final[df_final['Segmento'] == 'En Peligro']), delta_color="inverse")
c4.metric("Socios Estratégicos", len(df_final[df_final['Segmento'] == 'Socio Estratégico']))

col_L, col_R = st.columns([2,1])

with col_L:
    st.subheader("Mapa de Calor (Matriz RFM)")
    # Pivotar y reindexar para evitar errores
    rfm_agg = df_final.groupby(['R_Score', 'F_Score']).size().reset_index(name='Conteo')
    rfm_pivot = rfm_agg.pivot(index='F_Score', columns='R_Score', values='Conteo').fillna(0)
    all_scores = [1, 2, 3, 4, 5]
    rfm_pivot = rfm_pivot.reindex(index=all_scores, columns=all_scores, fill_value=0).sort_index(ascending=False)
    
    fig_heat = px.imshow(rfm_pivot, 
        labels=dict(x="Recencia (5=Hoy)", y="Frecuencia (5=Siempre)", color="Clientes"),
        x=['1 (Perdido)', '2', '3', '4', '5 (Activo)'], 
        y=['5 (Muy Frecuente)', '4', '3', '2', '1 (Baja)'], 
        text_auto=True, color_continuous_scale='RdYlGn')
    st.plotly_chart(fig_heat, use_container_width=True)

with col_R:
    st.subheader("Segmentación Actual")
    fig_pie = px.pie(df_final, names='Segmento', hole=0.5, color='Segmento', 
                     color_discrete_map={'En Peligro':'#e74c3c', 'Socio Estratégico':'#2ecc71', 'Cliente Regular':'#f1c40f', 'Ocasional': '#95a5a6', 'Bajo Potencial': '#34495e'})
    st.plotly_chart(fig_pie, use_container_width=True)

# ==========================================
# FASE 2: INTELIGENCIA ARTIFICIAL (NUEVO)
# ==========================================
st.markdown("---")
st.header("Inteligencia Artificial (Predictivo)")
st.markdown("Algoritmos avanzados para detectar patrones ocultos y riesgo futuro.")

tab_cluster, tab_pred = st.tabs(["Clustering (K-Means)", "Predicción de Fuga"])

with tab_cluster:
    col_c1, col_c2 = st.columns([2, 1])
    with col_c1:
        st.markdown("#### Agrupación Matemática en 3D")
        st.caption("La IA agrupa clientes por similitud de comportamiento (Recencia, Frecuencia, Venta).")
        fig_3d = px.scatter_3d(df_final, x='Recencia_Dias', y='Frecuencia_Mensual', z='Venta_Mensual',
                               color='Cluster_IA', opacity=0.8, size_max=10,
                               color_discrete_sequence=px.colors.qualitative.Bold)
        fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=400)
        st.plotly_chart(fig_3d, use_container_width=True)
    with col_c2:
        st.markdown("#### Clusters Detectados")
        conteo_cluster = df_final['Cluster_IA'].value_counts().reset_index()
        conteo_cluster.columns = ['Cluster', 'Cantidad']
        st.dataframe(conteo_cluster, use_container_width=True, hide_index=True)

with tab_pred:
    st.markdown("#### Probabilidad de Fuga (Mes Siguiente)")
    st.caption("Score calculado por modelo probabilístico (0% a 100%).")
    
    # Histograma de Riesgo
    fig_hist = px.histogram(df_final, x="Prob_Fuga_ML", nbins=20, 
                            title="Distribución del Riesgo en la Cartera",
                            labels={'Prob_Fuga_ML': 'Probabilidad de Fuga (0 a 1)'},
                            color_discrete_sequence=['#FF4B4B'])
    st.plotly_chart(fig_hist, use_container_width=True)
    
    st.markdown("**Top 5 Clientes en Riesgo Crítico (IA):**")
    st.dataframe(df_final[['Nombre_Tienda', 'Prob_Fuga_ML', 'Cluster_IA', 'Segmento']]
                 .sort_values('Prob_Fuga_ML', ascending=False).head(5)
                 .style.format({'Prob_Fuga_ML': '{:.1%}'}), use_container_width=True)

# ==========================================
# FASE 3: LABORATORIO DE ACCIÓN (LO QUE YA TENÍAS)
# ==========================================
st.markdown("---")
st.header("Laboratorio de Impacto (Prescriptivo)")

tab_sim, tab_geo, tab_bot = st.tabs(["Simulador ROI", "Geo-Oportunidades", "Acción WhatsApp"])

with tab_sim:
    st.info("**Lógica Financiera:** ROI basado en recuperar **1 MES** de venta futura, descontando costos logísticos y descuentos.")
    
    col_input, col_kpi = st.columns([1, 2])
    
    # Datos target (Usando 'En Peligro' para mantener consistencia con lo anterior)
    target_df = df_final[df_final['Segmento'] == 'En Peligro']
    n_target = len(target_df)
    venta_mensual_riesgo = target_df['Venta_Mensual'].sum()
    
    with col_input:
        st.markdown("**Inversión**")
        costo_visita = st.number_input("Costo Visita ($)", value=8000, step=1000)
        descuento = st.slider("Descuento Gancho (%)", 0, 20, 5)
        st.markdown("**Retorno Esperado**")
        efectividad = st.slider("Tasa Recuperación (%)", 10, 60, 20)
        margen_bruto = st.slider("Margen Producto (%)", 5, 30, 15) / 100

    with col_kpi:
        # CÁLCULOS (ROI CORREGIDO)
        clientes_recuperados = int(n_target * (efectividad / 100))
        venta_recuperada_bruta = (venta_mensual_riesgo / n_target) * clientes_recuperados if n_target > 0 else 0
        utilidad_bruta = venta_recuperada_bruta * margen_bruto
        
        gasto_operativo = n_target * costo_visita 
        costo_descuento = venta_recuperada_bruta * (descuento / 100)
        inversion_total = gasto_operativo + costo_descuento
        
        profit_neto = utilidad_bruta - inversion_total
        roi = (profit_neto / inversion_total) * 100 if inversion_total > 0 else 0
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Inversión Total", f"${inversion_total:,.0f}")
        m2.metric("Utilidad (1 Mes)", f"${utilidad_bruta:,.0f}")
        m3.metric("ROI Real", f"{roi:.1f}%", delta=f"${profit_neto:,.0f} Neto", delta_color="normal" if roi > 0 else "inverse")
        
        fig_wf = go.Figure(go.Waterfall(
            orientation = "v", measure = ["relative", "relative", "total"],
            x = ["Inversión", "Margen Generado", "Resultado"],
            text = [f"-${inversion_total/1000:.0f}k", f"+${utilidad_bruta/1000:.0f}k", f"${profit_neto/1000:.0f}k"],
            y = [-inversion_total, utilidad_bruta, profit_neto],
            connector = {"line":{"color":"rgb(63, 63, 63)"}},
            decreasing = {"marker":{"color":"#ef553b"}}, increasing = {"marker":{"color":"#00cc96"}}, totals = {"marker":{"color":"#636efa"}}
        ))
        fig_wf.update_layout(title="Rentabilidad de la Campaña", height=300)
        st.plotly_chart(fig_wf, use_container_width=True)

with tab_geo:
    st.subheader("Geolocalización")
    filtro = st.radio("Objetivo:", ["Recuperar (En Peligro)", "Fidelizar (Socio Estratégico)"], horizontal=True)
    cat_filtro = "En Peligro" if "Recuperar" in filtro else "Socio Estratégico"
    df_map = df_final[df_final['Segmento'] == cat_filtro]
    if not df_map.empty:
        st.map(df_map, latitude='lat', longitude='lon', size=20, color='#ff4b4b' if "Peligro" in filtro else '#00cc96')
    else:
        st.warning("No hay datos.")

with tab_bot:
    st.subheader("Generador Next Best Action")
    tienda = st.selectbox("Buscar Tienda:", df_final['Nombre_Tienda'].head(15))
    
    if tienda:
        info = df_final[df_final['Nombre_Tienda'] == tienda].iloc[0]
        
        # Lógica enriquecida con la fase predictiva
        cluster_ia = info['Cluster_IA']
        prob_fuga = info['Prob_Fuga_ML']
        
        if info['Segmento'] == 'En Peligro':
            msg = f"Hola *{tienda}*, te extrañamos. Detectamos que estás en nuestro cluster '{cluster_ia}' y queremos verte volver. Tienes **5% OFF** hoy."
            tipo = "RECUPERACIÓN"
            color = "error"
        elif cluster_ia == "Cuentas AAA":
            msg = f"¡Hola *{tienda}*! Como eres una de nuestras 'Cuentas AAA', tienes prioridad de despacho mañana."
            tipo = "FIDELIZACIÓN VIP"
            color = "success"
        else:
            msg = f"Hola *{tienda}*, recuerda revisar el catálogo."
            tipo = "MANTENIMIENTO"
            color = "info"
            
        if prob_fuga > 0.7:
            st.warning(f"¡Cuidado! Este cliente tiene una Probabilidad de Fuga del {prob_fuga:.1%}. Se recomienda llamar.")
        
        if color == "error": st.error(f"Estrategia: {tipo}")
        elif color == "success": st.success(f"Estrategia: {tipo}")
        else: st.info(f"Estrategia: {tipo}")
        
        st.text_area("Mensaje:", value=msg, height=100)
        st.button("Enviar WhatsApp")
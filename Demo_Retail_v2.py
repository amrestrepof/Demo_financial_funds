import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="RutaFresca - Shopper Intelligence", layout="wide", page_icon="🚚")

# --- 1. GENERACIÓN DE DATOS (NUEVOS PERFILES & GEO-ZONAS) ---
@st.cache_data
def generar_datos(n_clientes):
    np.random.seed(42)
    ids = [f"CTE-{i:04d}" for i in range(n_clientes)]
    nombres = [f"Tienda {np.random.choice(['El Vecino','La Esperanza','Don Pepe','La Esquina','Doña Rosa','Santa Maria','El Progreso'])} {i}" for i in range(n_clientes)]
    
    # --- DEFINICIÓN DE PERFILES DE SHOPPER ---
    perfiles_shopper = [
        'Madre Ejecutiva (Tiempo)',       # Busca practicidad, ticket alto, baja frecuencia
        'Mujer Independiente (Fit/Gourmet)', # Busca nicho, saludable, porciones pequeñas
        'Admin. Hogar (Tradicional)',     # Busca rendimiento, economía, volumen
        'Estudiantes/De Paso',            # Snacks, bebidas
        'Mixto'
    ]
    
    # --- SIMULACIÓN DE BARRIOS (ZONAS LÓGICAS) ---
    centros = {
        'Norte (Moderno)': {
            'lat': 4.71, 'lon': -74.05, 
            'prob': [0.40, 0.30, 0.10, 0.10, 0.10] # Predomina Ejecutiva e Independiente
        },
        'Centro (Urbano)': {
            'lat': 4.60, 'lon': -74.08, 
            'prob': [0.10, 0.20, 0.10, 0.50, 0.10] # Predomina Estudiante/Paso
        },
        'Sur/Oeste (Familiar)': {
            'lat': 4.55, 'lon': -74.12, 
            'prob': [0.15, 0.05, 0.60, 0.10, 0.10] # Predomina Admin Hogar Tradicional
        }
    }
    
    lats, lons, shoppers, zonas = [], [], [], []
    
    for _ in range(n_clientes):
        # Asignar a una zona
        zona_nombre = np.random.choice(list(centros.keys()))
        info_zona = centros[zona_nombre]
        
        # Generar coordenadas con dispersión
        lats.append(np.random.normal(info_zona['lat'], 0.02))
        lons.append(np.random.normal(info_zona['lon'], 0.02))
        zonas.append(zona_nombre)
        # Asignar Shopper según probabilidad de la zona
        shoppers.append(np.random.choice(perfiles_shopper, p=info_zona['prob']))

    # --- DATOS TRANSACCIONALES ---
    recencia = np.random.randint(1, 90, n_clientes) 
    frecuencia_mensual = np.random.randint(1, 10, n_clientes) 
    # Ticket base varía un poco (ruido)
    ticket_base = np.random.uniform(50, 400, n_clientes) * 1000 
    
    venta_mensual = ticket_base * frecuencia_mensual
    venta_semestral = venta_mensual * 6

    # --- DATOS SOCIODEMOGRÁFICOS TENDERO ---
    genero_tendero = np.random.choice(['Femenino', 'Masculino'], n_clientes, p=[0.70, 0.30])
    edad_tendero = np.random.randint(22, 68, n_clientes)

    df = pd.DataFrame({
        'Cliente_ID': ids, 'Nombre_Tienda': nombres,
        'Recencia_Dias': recencia, 'Frecuencia_Mensual': frecuencia_mensual,
        'Venta_Mensual': venta_mensual, 'Venta_Total': venta_semestral,
        'Genero_Tendero': genero_tendero, 'Edad_Tendero': edad_tendero,
        'Shopper_Predominante': shoppers, 'Zona_Barrio': zonas,
        'lat': pd.Series(lats), 'lon': pd.Series(lons)
    })
    return df

# --- 2. CEREBRO ANALÍTICO ---
def procesar_inteligencia(df):
    # A. RFM & ILC SCORE
    df['R_Score'] = pd.qcut(df['Recencia_Dias'], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    # Rank para evitar error si hay muchos repetidos
    df['F_Score'] = (df['Frecuencia_Mensual'].rank(pct=True, method='first') * 5).apply(np.ceil).astype(int)
    df['M_Score'] = pd.qcut(df['Venta_Total'], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    
    df['ILC_Score'] = ((df['R_Score']*0.3 + df['F_Score']*0.3 + df['M_Score']*0.4) / 5) * 100
    df['ILC_Score'] = df['ILC_Score'].round(1)

    # Segmentación Clásica
    def asignar_segmento(row):
        if row['Recencia_Dias'] > 45: return 'En Peligro'
        if row['ILC_Score'] >= 85: return 'Socio Estratégico'
        if row['ILC_Score'] >= 60: return 'Cliente Regular'
        return 'Bajo Potencial'
    df['Segmento'] = df.apply(asignar_segmento, axis=1)

    # B. CLUSTERING IA (K-MEANS)
    scaler = MinMaxScaler()
    features = df[['Recencia_Dias', 'Frecuencia_Mensual', 'Venta_Mensual']]
    features_scaled = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Cluster_ID'] = kmeans.fit_predict(features_scaled)
    
    # Nombres Corporativos
    cluster_map = {0: "Recurrentes", 1: "Cuentas AAA", 2: "Baja Rotación", 3: "En Desarrollo"}
    df['Cluster_IA'] = df['Cluster_ID'].map(cluster_map).fillna("Otros")

    # C. PREDICCIÓN FUGA (ML SIMULADO)
    raw_prob = (df['Recencia_Dias'] * 0.7) - (df['Frecuencia_Mensual'] * 5)
    df['Prob_Fuga_ML'] = (raw_prob - raw_prob.min()) / (raw_prob.max() - raw_prob.min())
    
    return df

# --- 3. INTERFAZ GRÁFICA ---

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2821/2821828.png", width=80)
st.sidebar.title(" RutaFresca")
st.sidebar.caption("Shopper Intelligence 360°")
n_clientes = st.sidebar.slider("Tamaño Base de Datos", 200, 2000, 800)
st.sidebar.markdown("---")
# Filtros
filtro_perfil = st.sidebar.multiselect("Filtrar por Perfil Shopper", 
                                       ['Madre Ejecutiva (Tiempo)', 'Mujer Independiente (Fit/Gourmet)', 'Admin. Hogar (Tradicional)', 'Estudiantes/De Paso'],
                                       default=['Madre Ejecutiva (Tiempo)', 'Admin. Hogar (Tradicional)'])

# Procesamiento
raw_df = generar_datos(n_clientes)
df_processed = procesar_inteligencia(raw_df)

# Aplicar Filtro
if filtro_perfil:
    df_final = df_processed[df_processed['Shopper_Predominante'].isin(filtro_perfil)]
else:
    df_final = df_processed

st.title("Tablero de Inteligencia Comercial")

# EXPLICACIÓN TÉCNICA
with st.expander(" CÓMO FUNCIONA: Arquitectura de Inteligencia", expanded=False):
    st.markdown("""
    1.  **Datos 360:** Integramos transacciones con perfiles de **Shopper** (Ejecutiva, Ama de Casa, etc.) y **Geo-Entorno**.
    2.  **Algoritmos:** * *ILC Score:* Reglas de negocio para medir lealtad.
        * *Clustering K-Means:* IA para detectar grupos (Cuentas AAA, Recurrentes).
    3.  **Acción:** Simulador ROI y Bot Contextual.
    """)

# KPI Section
c1, c2, c3, c4 = st.columns(4)
c1.metric("Tiendas Filtradas", len(df_final))
c2.metric("Venta Promedio", f"${df_final['Venta_Mensual'].mean():,.0f}")
c3.metric("Shopper Dominante", df_final['Shopper_Predominante'].mode()[0] if not df_final.empty else "N/A")
c4.metric("Cuentas AAA (IA)", len(df_final[df_final['Cluster_IA'] == 'Cuentas AAA']))

st.markdown("---")

# TABS PRINCIPALES (ORGANIZACIÓN COMPLETA)
tab1, tab2, tab3, tab4 = st.tabs(["1️⃣ Diagnóstico RFM", "2️⃣ Fokus Mujer & Shopper", "3️⃣ Geo-Entorno", "4️⃣ Estrategia & ROI"])

# --- TAB 1: DIAGNÓSTICO ---
with tab1:
    col_L, col_R = st.columns([2,1])
    with col_L:
        st.subheader("Matriz de Calor (Retención)")
        # Pivot seguro con reindex
        rfm_agg = df_final.groupby(['R_Score', 'F_Score']).size().reset_index(name='Conteo')
        rfm_pivot = rfm_agg.pivot(index='F_Score', columns='R_Score', values='Conteo').fillna(0)
        rfm_pivot = rfm_pivot.reindex(index=[1,2,3,4,5], columns=[1,2,3,4,5], fill_value=0).sort_index(ascending=False)
        
        fig_heat = px.imshow(rfm_pivot, labels=dict(x="Recencia", y="Frecuencia"), 
                             x=['Perdido', 'Riesgo', 'Dormido', 'Activo', 'Nuevo'],
                             y=['Muy Frecuente', 'Frecuente', 'Regular', 'Baja', 'Muy Baja'],
                             text_auto=True, color_continuous_scale='RdYlGn')
        st.plotly_chart(fig_heat, use_container_width=True)
    with col_R:
        st.subheader("Segmentación IA")
        fig_pie = px.pie(df_final, names='Cluster_IA', hole=0.5, color='Cluster_IA',
                         color_discrete_map={' Baja Rotación':'#e74c3c', ' Cuentas AAA':'#2ecc71', 'Recurrentes':'#f1c40f'})
        st.plotly_chart(fig_pie, use_container_width=True)

# --- TAB 2: FOKUS MUJER & SHOPPER (NUEVO) ---
with tab2:
    st.subheader("Perfilamiento del Consumidor Final")
    st.caption("Entendiendo los nuevos roles de la mujer en la compra.")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.markdown("**Gasto Mensual por Tipo de Shopper**")
        fig_box = px.box(df_processed, x="Shopper_Predominante", y="Venta_Mensual", color="Shopper_Predominante",
                         color_discrete_map={
                             'Madre Ejecutiva (Tiempo)': '#9b59b6',
                             'Mujer Independiente (Fit/Gourmet)': '#1abc9c',
                             'Admin. Hogar (Tradicional)': '#e84393',
                             'Estudiantes/De Paso': '#f1c40f'
                         })
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)
    
    with col_s2:
        st.markdown("**Insight Demográfico (Tendero)**")
        fig_age = px.histogram(df_final, x="Edad_Tendero", color="Genero_Tendero", nbins=15, 
                               color_discrete_map={'Femenino':'#e84393', 'Masculino':'#74b9ff'},
                               title="Edad y Género del Dueño de Tienda")
        st.plotly_chart(fig_age, use_container_width=True)

# --- TAB 3: GEO-ENTORNO (ZONAS) ---
with tab3:
    st.header(" Mapa de Oportunidades")
    st.markdown("Distribución espacial según el perfil de consumo dominante.")
    
    col_map, col_zn = st.columns([2, 1])
    with col_map:
        fig_map = px.scatter_mapbox(df_processed, lat="lat", lon="lon", color="Shopper_Predominante",
                                    size="Venta_Mensual", hover_name="Nombre_Tienda",
                                    zoom=11, height=500, mapbox_style="carto-positron",
                                    color_discrete_map={
                                        'Madre Ejecutiva (Tiempo)': '#9b59b6',
                                        'Mujer Independiente (Fit/Gourmet)': '#1abc9c',
                                        'Admin. Hogar (Tradicional)': '#e84393',
                                        'Estudiantes/De Paso': '#f1c40f',
                                        'Mixto': '#95a5a6'
                                    })
        st.plotly_chart(fig_map, use_container_width=True)
    
    with col_zn:
        st.markdown("### Estrategia por Zona")
        st.info("🟣 **Norte (Ejecutiva):** Vender 'Tiempo'. Packs de Lonchera y Congelados.")
        st.success("🟢 **Centro (Independiente):** Vender 'Bienestar'. Snacks Fit y Bebidas.")
        st.error("🔴 **Sur (Tradicional):** Vender 'Ahorro'. Formatos Familiares (Arroz 5kg).")

# --- TAB 4: ESTRATEGIA & ACCIÓN ---
with tab4:
    st.header("Laboratorio de Impacto")
    
    # 1. CLUSTERING 3D
    with st.expander("Ver Clusterización IA en 3D", expanded=False):
        fig_3d = px.scatter_3d(df_final, x='Recencia_Dias', y='Frecuencia_Mensual', z='Venta_Mensual',
                               color='Cluster_IA', opacity=0.8, size_max=10,
                               color_discrete_sequence=px.colors.qualitative.Bold)
        fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0), height=400)
        st.plotly_chart(fig_3d, use_container_width=True)

    col_sim, col_bot = st.columns(2)

    # 2. SIMULADOR ROI
    with col_sim:
        st.subheader(" Simulador ROI")
        st.caption("Cálculo sobre 1 mes de recuperación.")
        
        n_target = len(df_final)
        venta_base = df_final['Venta_Mensual'].sum()
        
        inv_visita = st.number_input("Costo Logístico Total ($)", value=n_target * 5000)
        tasa_conv = st.slider("Conversión Esperada (%)", 5, 50, 20) / 100
        margen = 0.15 # 15%
        
        venta_recup = venta_base * tasa_conv
        utilidad = venta_recup * margen
        roi = ((utilidad - inv_visita) / inv_visita) * 100 if inv_visita > 0 else 0
        
        st.metric("Inversión", f"${inv_visita:,.0f}")
        st.metric("Utilidad Generada", f"${utilidad:,.0f}")
        st.metric("ROI", f"{roi:.1f}%", delta_color="normal" if roi > 0 else "inverse")

    # 3. BOT WHATSAPP CONTEXTUAL
    with col_bot:
        st.subheader("Bot Contextual")
        st.caption("El mensaje se adapta al Shopper de la tienda.")
        
        tienda_sel = st.selectbox("Elegir Tienda:", df_final['Nombre_Tienda'].head(15))
        
        if tienda_sel:
            dat = df_final[df_final['Nombre_Tienda'] == tienda_sel].iloc[0]
            shopper = dat['Shopper_Predominante']
            
            # Lógica de Script Dinámico
            if shopper == 'Madre Ejecutiva (Tiempo)':
                txt = "Sabemos que tus clientas corren todo el día. Ofréceles el 'Kit Lonchera' listo para llevar."
                style = "info"
            elif shopper == 'Mujer Independiente (Fit/Gourmet)':
                txt = "Tus clientas buscan cuidarse. Envíales los nuevos Yogures Griegos y Snacks de Arroz."
                style = "success"
            elif shopper == 'Admin. Hogar (Tradicional)':
                txt = "Es momento de surtir Aceite y Granos en gran formato. Ellas buscan rendimiento."
                style = "error"
            else:
                txt = "Recuerda surtir los productos de alta rotación."
                style = "warning"
            
            msg_final = f"Hola {tienda_sel}. {txt}"
            
            if style == 'info': st.info(f"Shopper: {shopper}")
            elif style == 'success': st.success(f"Shopper: {shopper}")
            elif style == 'error': st.error(f"Shopper: {shopper}")
            else: st.warning(f"Shopper: {shopper}")
            
            st.text_area("Mensaje Generado:", value=msg_final, height=80)
            st.button("Enviar WhatsApp ")
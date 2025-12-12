import streamlit as st
import pandas as pd
import numpy as np
import time
import random
import re
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Intelli-PQR Analytics Suite",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- ESTILOS CSS ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    h1, h2, h3 { color: #2c3e50; }
    </style>
    """, unsafe_allow_html=True)

# --- GENERADOR DE DATOS SIMULADOS (MOCKUP DATA) ---
@st.cache_data
def generar_datos_historicos():
    """Genera un DataFrame de 6 meses de operación simulada"""
    fechas = pd.date_range(end=datetime.today(), periods=180)
    data = []
    
    for fecha in fechas:
        # Simular volumen diario (entre 50 y 150 casos)
        volumen = np.random.randint(50, 150)
        for _ in range(volumen):
            cat = np.random.choice(
                ["Falla Técnica", "Facturación", "Retención", "Info General"],
                p=[0.4, 0.3, 0.1, 0.2]
            )
            
            # El churn score depende de la categoría
            churn_base = 80 if cat == "Retención" else (60 if cat == "Falla Técnica" else 20)
            churn_score = min(100, max(0, int(np.random.normal(churn_base, 15))))
            
            # Tiempos de atención (AI es rápido, Humano es lento)
            tiempo_ai = np.random.uniform(0.1, 0.5) # segundos
            tiempo_humano = np.random.uniform(5, 45) # minutos
            
            data.append({
                "Fecha": fecha,
                "Categoría": cat,
                "Churn_Score": churn_score,
                "Tiempo_Resolucion_Min": tiempo_humano if np.random.random() > 0.8 else tiempo_ai, # 80% automatizado
                "Canal": np.random.choice(["Email", "Web", "Call Center"]),
                "Estado": "Cerrado" if np.random.random() > 0.1 else "Abierto",
                "Ingresos_Protegidos": np.random.randint(50000, 300000) if churn_score > 70 else 0
            })
            
    return pd.DataFrame(data)

df_historico = generar_datos_historicos()

# --- FUNCIONES DE LÓGICA DE NEGOCIO (MODELOS) ---
# (Mismas funciones lógicas del demo anterior, optimizadas)

def modelo_1_clasificacion(texto):
    texto = texto.lower()
    if "factura" in texto or "cobro" in texto: return "Facturación y Cobranza", 0.98, "Revisión Automática"
    elif "señal" in texto or "internet" in texto: return "Soporte Técnico / Falla de Red", 0.95, "Soporte Nivel 2"
    elif "cancelar" in texto or "retiro" in texto: return "Retención / Fidelización", 0.89, "Equipo de Retención"
    else: return "Otros / Información General", 0.65, "Mesa de Ayuda (Humano)"

def modelo_2_ner(texto):
    entidades = []
    cedulas = re.findall(r'\b\d{1,3}[.]?\d{3}[.]?\d{3}\b|\b\d{7,10}\b', texto)
    if cedulas: entidades.append({"Entidad": "ID Cliente", "Valor": cedulas[0]})
    cun = re.findall(r'CUN-\d+', texto)
    if cun: entidades.append({"Entidad": "Código CUN", "Valor": cun[0]})
    if not entidades: entidades.append({"Entidad": "Aviso", "Valor": "Se requiere validación manual"})
    return pd.DataFrame(entidades)

def modelo_3_churn(categoria):
    if "Retención" in categoria: return random.randint(85, 99), "CRÍTICO"
    elif "Falla" in categoria: return random.randint(50, 80), "ALTO"
    else: return random.randint(10, 40), "BAJO"

# --- PESTAÑA 1: DASHBOARD GERENCIAL ---
def mostrar_dashboard():
    st.title("Tablero de Control Gerencial")
    st.markdown("Visión estratégica del impacto de la solución analítica en el negocio.")
    
    # Bloque de explicación general
    with st.expander("¿Qué estoy viendo aquí? (Guía para el Demo)"):
        st.write("""
        Este tablero simula los resultados acumulados de 6 meses de operación. 
        Permite medir el retorno de inversión (ROI) a través de la reducción de fuga de clientes 
        y la optimización del tiempo de los empleados.
        """)
    
    st.markdown("---")

    # --- SECCIÓN A: KPIS DE ALTO NIVEL ---
    st.subheader("1. Indicadores Clave de Desempeño (KPIs)")
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    # Cálculos
    total_casos = len(df_historico)
    automatizacion = (len(df_historico[df_historico['Tiempo_Resolucion_Min'] < 1]) / total_casos) * 100
    dinero_salvado = df_historico['Ingresos_Protegidos'].sum()
    churn_promedio = df_historico['Churn_Score'].mean()

    # Métricas con Tooltips (Pasa el mouse por encima del signo ?)
    kpi1.metric(
        label="Total PQRs", 
        value=f"{total_casos:,}", 
        delta="+12% vs mes ant",
        help="Volumen total de interacciones (correos, llamadas, web) procesadas por el modelo en el periodo."
    )
    
    kpi2.metric(
        label="Automatización", 
        value=f"{automatizacion:.1f}%", 
        delta="Objetivo: 80%",
        help="Porcentaje de casos donde la IA clasificó, extrajo datos y enrutó sin intervención humana."
    )
    
    kpi3.metric(
        label="Ingresos Protegidos", 
        value=f"${dinero_salvado:,.0f}", 
        delta="Retención Exitosa",
        help="Suma de la facturación de clientes con 'Riesgo Crítico' que fueron detectados a tiempo y NO cancelaron el servicio."
    )
    
    kpi4.metric(
        label="Riesgo Promedio", 
        value=f"{churn_promedio:.1f}/100", 
        delta="-2.5 pts (Mejora)",
        delta_color="inverse",
        help="Puntaje promedio de probabilidad de fuga de la base de clientes. Menor es mejor."
    )

    st.markdown("---")

    # --- SECCIÓN B: ANÁLISIS DE CAUSA Y TENDENCIA ---
    st.subheader("2. Radiografía del Negocio")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("**Motivos de Contacto (Clasificación Semántica)**")
        fig_pie = px.pie(df_historico, names='Categoría', hole=0.4, 
                         color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig_pie, use_container_width=True)
        
        with st.expander("Análisis del Gráfico"):
            st.info("""
            **¿Qué nos dice el Modelo 1?**
            Muestra las verdaderas razones de las quejas entendiendo el contexto, no solo palabras clave.
            *Si el segmento 'Falla Técnica' crece, alerta al área de ingeniería.*
            """)
    
    with c2:
        st.markdown("**Evolución del Riesgo de Fuga (Churn)**")
        df_diario = df_historico.groupby('Fecha')[['Churn_Score']].mean().reset_index()
        fig_line = px.line(df_diario, x='Fecha', y='Churn_Score', line_shape='spline')
        fig_line.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Umbral de Alerta")
        fig_line.update_traces(line_color='#e74c3c')
        st.plotly_chart(fig_line, use_container_width=True)
        
        with st.expander("Análisis del Gráfico"):
            st.info("""
            **¿Qué nos dice el Modelo 3?**
            Monitorea el 'temperamento' de los clientes. 
            *Picos por encima de la línea roja indican momentos de crisis operativa que podrían causar retiros masivos.*
            """)

    st.markdown("---")

    # --- SECCIÓN C: COMPARATIVA DE EFICIENCIA ---
    st.subheader("3. Impacto Operativo: IA vs Humano")
    
    fig_hist = px.histogram(df_historico, x="Tiempo_Resolucion_Min", color="Categoría", nbins=50, 
                            labels={'Tiempo_Resolucion_Min':'Minutos de Gestión'},
                            log_y=True,
                            color_discrete_sequence=px.colors.qualitative.Pastel)
    
    # Anotaciones para señalar las áreas del gráfico
    fig_hist.add_vrect(x0=0, x1=1, annotation_text="Zona IA (<1 min)", annotation_position="top left", fillcolor="green", opacity=0.1, line_width=0)
    fig_hist.add_vrect(x0=5, x1=50, annotation_text="Zona Humana (>5 min)", annotation_position="top right", fillcolor="red", opacity=0.1, line_width=0)
    
    st.plotly_chart(fig_hist, use_container_width=True)
    
    with st.expander("Por qué este gráfico cierra ventas"):
        st.write("""
        **Interpretación:**
        * **Izquierda (Zona Verde):** Miles de casos resueltos en segundos por los modelos. Costo marginal cercano a cero.
        * **Derecha (Zona Roja):** La larga cola de trabajo manual. 
        
        **Conclusión:** La solución elimina la carga operativa repetitiva, dejando a los humanos solo los casos complejos.
        """)

# --- PESTAÑA 2: DEMO EN VIVO ---
def mostrar_inferencia():
    st.title("Centro de Operaciones (Tiempo Real)")
    st.markdown("Simulación del flujo de trabajo de un agente asistido por IA.")
    
    # Selector de caso
    caso = st.selectbox("Seleccionar caso de prueba:", 
        ["Caso A: Facturación (Riesgo Bajo)", "Caso B: Falla Técnica (Riesgo Medio)", "Caso C: Deserción (Riesgo Crítico)"])
    
    # Textos precargados
    textos = {
        "Caso A: Facturación (Riesgo Bajo)": "Hola, me llegó la factura por un valor más alto. Mi CC es 80.123.456. Quiero que revisen.",
        "Caso B: Falla Técnica (Riesgo Medio)": "Llevo 2 días sin internet en la casa. He reiniciado el modem y nada. Mi CUN anterior es CUN-556677.",
        "Caso C: Deserción (Riesgo Crítico)": "Voy a cancelar el servicio. Claro me ofrece más gigas. Llevo 10 años con ustedes y el servicio es pésimo. CC 79.999.000."
    }
    
    col_in, col_res = st.columns([1, 1])
    
    with col_in:
        st.subheader("Entrada")
        txt = st.text_area("Mensaje del Cliente", value=textos[caso], height=200)
        btn = st.button("Analizar PQR", type="primary")

    if btn:
        with st.spinner('Ejecutando Modelos de NLP y Predicción...'):
            time.sleep(1.5) # Drama
            
            cat, conf, ruta = modelo_1_clasificacion(txt)
            df_ner = modelo_2_ner(txt)
            score, nivel = modelo_3_churn(cat)
            
        with col_res:
            st.subheader("Salida de Modelos")
            
            # Tarjetas de resultados
            st.info(f"**Clasificación:** {cat} ({conf*100:.0f}%)")
            
            if nivel == "CRÍTICO":
                st.error(f" **Riesgo Churn:** {score}/100 ({nivel})")
            elif nivel == "ALTO":
                st.warning(f" **Riesgo Churn:** {score}/100 ({nivel})")
            else:
                st.success(f" **Riesgo Churn:** {score}/100 ({nivel})")
                
            st.write("**Datos Extraídos:**")
            st.dataframe(df_ner, hide_index=True)
            
            st.markdown("---")
            st.markdown("**Acción Sugerida:**")
            if nivel == "CRÍTICO":
                st.write(" **Transferir a Retención Inmediata + Oferta 20% OFF**")
            else:
                st.write(" **Generar Ticket Automático y enviar a cola**")

# --- NAVEGACIÓN PRINCIPAL ---
#st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3048/3048122.png", width=80)
st.sidebar.title("Navegación")
modo = st.sidebar.radio("Ir a:", [" Dashboard Gerencial", " Demo Tiempo Real"])

st.sidebar.markdown("---")
st.sidebar.info("""
**Arquitectura:**
1. Clasificador Semántico
2. Extractor NER
3. Predictor Churn
""")

if modo == " Dashboard Gerencial":
    mostrar_dashboard()
else:
    mostrar_inferencia()
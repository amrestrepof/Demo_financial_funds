import streamlit as st
import pandas as pd

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="CreditGas - Demo Analítico", layout="wide", initial_sidebar_state="expanded")

# --- ESTILOS PERSONALIZADOS ---
st.markdown("""
    <style>
    .metric-card { background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 5px solid #004aad; }
    .recom-card { background-color: #e8f4f8; padding: 15px; border-radius: 10px; border-left: 5px solid #00b4d8; }
    </style>
    """, unsafe_allow_html=True)

# --- DATOS SIMULADOS (BASE DE DATOS HISTÓRICA) ---
@st.cache_data
def cargar_datos_clientes():
    return pd.DataFrame({
        'ID': ['C001', 'C002', 'C003', 'C004'],
        'Nombre': ['Ana Silva', 'Carlos Gómez', 'María López', 'Juan Pérez'],
        'Estrato': [4, 2, 3, 5],
        'Antiguedad_anios': [5, 1, 10, 2],
        'Promedio_Recibo': [45000, 15000, 30000, 85000],
        'Dias_Ultimo_Pago': [12, 65, 5, 2],     # R: Recencia (Días de mora)
        'Pagos_Ultimo_Anio': [12, 6, 12, 11],   # F: Frecuencia (Pagos en el año)
        'Gasto_Historico': [3450000, 450000, 8500000, 1200000] # M: Monto histórico
    })

df_clientes = cargar_datos_clientes()

# --- SIDEBAR: CONTROLES DEL DEMO ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2043/2043102.png", width=50) # Icono genérico
st.sidebar.title("Configuración del Demo")
modo = st.sidebar.radio("📋 Modo de Operación", ["Analizar Cliente Histórico", "Simular Cliente Nuevo"])

st.sidebar.divider()

if modo == "Analizar Cliente Histórico":
    st.sidebar.subheader("Buscador de Clientes")
    cliente_sel = st.sidebar.selectbox("Seleccione un cliente:", df_clientes['Nombre'])
    datos = df_clientes[df_clientes['Nombre'] == cliente_sel].iloc[0]
    
    # Asignación de variables desde la base de datos
    nombre = datos['Nombre']
    estrato = datos['Estrato']
    antiguedad = datos['Antiguedad_anios']
    monto_recibo = datos['Promedio_Recibo']
    recencia = datos['Dias_Ultimo_Pago']
    frecuencia = datos['Pagos_Ultimo_Anio']
    monto_hist = datos['Gasto_Historico']
    
else:
    st.sidebar.subheader("Ingreso Manual de Datos")
    nombre = st.sidebar.text_input("Nombre Completo", "Nuevo Usuario")
    estrato = st.sidebar.slider("Estrato Socioeconómico", 1, 6, 3)
    antiguedad = st.sidebar.number_input("Años con el servicio", 0, 20, 1)
    monto_recibo = st.sidebar.number_input("Promedio mensual recibo ($)", 10000, 500000, 35000)
    pago_oportuno = st.sidebar.checkbox("¿Paga siempre a tiempo?", value=True)
    
    # Asignación de variables simuladas para RFM
    recencia = 5 if pago_oportuno else 90
    frecuencia = 12 if pago_oportuno else 4
    monto_hist = 0 # Al ser nuevo, no tiene historia de compras aliadas

# --- LÓGICA DEL MODELO ANALÍTICO ---
def calcular_score_y_cupo(estrato, antiguedad, recibo, recencia, frecuencia):
    # Regla de negocio estricta: Si tiene mora mayor a 60 días, se niega el cupo.
    if recencia > 60:
        return 0, "En Riesgo / Moroso"
    
    # Cálculo Base
    base = recibo * 10
    multiplicador_estrato = estrato * 1.5
    bono_antiguedad = antiguedad * 100000
    
    # Ajuste por Modelo RFM (Frecuencia de pagos castiga o premia)
    factor_comportamiento = min(frecuencia / 10, 1.2) # Max 20% de premio por pagar siempre
    
    cupo_total = (base + (base * multiplicador_estrato) + bono_antiguedad) * factor_comportamiento
    
    # Segmentación
    if factor_comportamiento >= 1.0 and antiguedad > 3:
        segmento = "Cliente Diamante 💎"
    elif factor_comportamiento >= 0.8:
        segmento = "Cliente Estándar ⭐"
    else:
        segmento = "Cliente en Observación ⚠️"
        
    return min(cupo_total, 15000000), segmento # Capado a 15 Millones

cupo_asignado, segmento_cliente = calcular_score_y_cupo(estrato, antiguedad, monto_recibo, recencia, frecuencia)

def generar_recomendacion(cupo, monto_hist, segmento):
    if cupo == 0:
        return "Refinanciación de deuda de gas, Micro-seguros familiares.", "Alerta"
    elif monto_hist > 5000000 or "Diamante" in segmento:
        return "Concesionarios de Motos (Cuota inicial), Computadores de Alta Gama, Viajes.", "Alto Valor"
    elif monto_hist > 1000000 or estrato >= 3:
        return "Remodelación (Homecenter), Línea Blanca (Neveras, Lavadoras), Smartphones.", "Consumo Medio"
    else:
        return "Mercado (Éxito), Electrodomésticos menores (Airfryer, Licuadoras), Ropa.", "Básicos"

recomendacion_texto, tipo_recom = generar_recomendacion(cupo_asignado, monto_hist, segmento_cliente)


# --- INTERFAZ PRINCIPAL (DASHBOARD) ---
st.title("CreditGas: Plataforma de Perfilamiento")
st.markdown("Plataforma de asignación de micro-créditos y ecosistema de aliados vía factura de gas.")

col1, col2 = st.columns([1.5, 1])

# COLUMNA IZQUIERDA: PERFIL Y RFM
with col1:
    st.subheader(f"Perfil Analítico: {nombre}")
    st.markdown(f"**Segmento Asignado:** {segmento_cliente}")
    
    # Métricas principales
    m1, m2, m3 = st.columns(3)
    m1.metric("Cupo Aprobado", f"${cupo_asignado:,.0f}")
    m2.metric("Promedio Recibo", f"${monto_recibo:,.0f}")
    m3.metric("Antigüedad", f"{antiguedad} Años")
    
    # Módulo RFM (Solo tiene sentido mostrarlo si es histórico, pero lo dejamos visual)
    st.markdown("### Modelo de Comportamiento (RFM)")
    rfm_c1, rfm_c2, rfm_c3 = st.columns(3)
    with rfm_c1:
        st.markdown(f"<div class='metric-card'><b>Recencia</b><br>Hace {recencia} días pagó</div>", unsafe_allow_html=True)
    with rfm_c2:
        st.markdown(f"<div class='metric-card'><b>Frecuencia</b><br>{frecuencia}/12 pagos al año</div>", unsafe_allow_html=True)
    with rfm_c3:
        st.markdown(f"<div class='metric-card'><b>Monto (Aliados)</b><br>${monto_hist:,.0f}</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown("### 💡 Motor de Recomendaciones Inteligente")
    st.markdown(f"<div class='recom-card'><b>Sugerencias basadas en su perfil ({tipo_recom}):</b><br> {recomendacion_texto}</div>", unsafe_allow_html=True)


# COLUMNA DERECHA: MARKETPLACE
with col2:
    st.subheader("🛒 Ecosistema de Aliados")
    
    aliados = {
        "Mercado Básico (Éxito)": {"precio": 350000, "icon": "🥖"},
        "Combo Electro (Éxito)": {"precio": 1200000, "icon": "📺"},
        "Remodelación (Homecenter)": {"precio": 2500000, "icon": "🏠"},
        "Smartphone Gama Alta": {"precio": 3500000, "icon": "📱"},
        "Moto 125cc (Auteco)": {"precio": 8500000, "icon": "🏍️"}
    }
    
    if cupo_asignado == 0:
        st.error("❌ Cliente actualmente no apto para compras en el ecosistema. Ofrecer refinanciación.")
    else:
        for item, info in aliados.items():
            with st.expander(f"{info['icon']} {item} - ${info['precio']:,.0f}", expanded=True):
                if cupo_asignado >= info['precio']:
                    st.success("Cupo Suficiente")
                    st.button(f"Generar Bono de Compra", key=item, use_container_width=True)
                else:
                    st.warning("Cupo Insuficiente")
                    st.button("Generar Bono", key=f"des_{item}", disabled=True, use_container_width=True)
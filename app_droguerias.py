import streamlit as st
import pandas as pd
import numpy as np # Necesario para la lógica del mapa y distancia
from datetime import datetime, timedelta 
# Importaciones de módulos locales (deben existir)
from data_simulator_droguerias import generate_data 
from model_crosssell import detectar_producto, generar_recomendaciones

# --- Configuración Inicial ---
st.set_page_config(layout="wide", page_title="Demo IA Contact Center Farmacias")

# Cargar datos simulados. AHORA INCLUYE df_farmacias
try:
    df_catalogo, df_compras, df_farmacias = generate_data()
except ValueError:
    st.error("Error al cargar datos. Asegúrese de que 'generate_data()' en data_simulator.py retorne 3 DataFrames (catalogo, compras, farmacias).")
    st.stop()
    
# --- Sección 1: Motor de Venta Cruzada y Sustitución ---
st.header("1.  Motor de Venta Cruzada y Sustitución (CrossSell IQ)")
st.subheader("Simulador de Asistencia a Agente en Tiempo Real")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("**Simulador de Transcripción de Llamada**")
    texto_llamada = st.text_area(
        "Voz del Cliente:",
        "Quiero la Aspirina 500mg, ¿la tienen disponible?", # Texto de ejemplo para activar stock/sustitución
        height=150
    )
    producto_detectado = detectar_producto(texto_llamada)

with col2:
    st.markdown("**Asistencia de Venta (Resultado IA)**")

    if producto_detectado:
        st.success(f"**Producto Solicitado (IA):** {producto_detectado}")

        recomendaciones = generar_recomendaciones(producto_detectado, df_catalogo)

        # AJUSTE: Mostrar Alerta de Stock y Sustitución Unificada
        if recomendaciones['Alerta_Stock']:
            st.error(" ¡ALERTA DE STOCK BAJO!") # Alerta visual fuerte
        
        st.warning(" **Motor de Sustituciones:**")
        # El mensaje incluye info de stock bajo y sugerencia, si aplica
        st.write(recomendaciones['Sustituto_Mensaje']) 

        st.info(" **Venta Cruzada (Cross-Sell IQ):**")
        st.write(f"Recomendar **{recomendaciones['Complemento']}** (Clientes que compran {producto_detectado} suelen comprar esto).")
    else:
        st.warning("Escuchando... Esperando detección de producto en la transcripción.")


st.markdown("---")


## 2. Modelo FRM con Geopriorización (FRM + L)

st.header("2.  Modelo FRM para Recompra y Retención")
st.markdown("Identifica proactivamente clientes **en riesgo** y asigna la **Farmacia más Cercana**.")

# --- LÓGICA DE FRM Y GEOLOCALIZACIÓN ---

# 1. Cálculo de Recencia, Frecuencia y Monto
df_clientes = df_compras.groupby('ID_Cliente').agg(
    Recencia=('Fecha_Compra', lambda x: (datetime.now() - x.max()).days),
    Frecuencia=('Fecha_Compra', 'count'),
    Monto_Promedio=('Monto', 'mean')
).reset_index()

# 2. Obtener Ubicación Promedio del Cliente
df_ubicacion_cliente = df_compras.groupby('ID_Cliente').agg(
    Latitud_Cliente=('Latitud_Cliente', 'mean'),
    Longitud_Cliente=('Longitud_Cliente', 'mean')
).reset_index()

df_clientes = df_clientes.merge(df_ubicacion_cliente, on='ID_Cliente', how='left')


# 3. Función para Segmentación y Prioridad (Semáforo)
def get_segmento_y_prioridad(row):
    R = row['Recencia']
    F = row['Frecuencia']
    
    if R < 30 and F >= 5:
        return 'Cliente VIP', '🟡 MEDIA (Fidelización)' 
    elif R > 90 and F < 2:
        return 'En Riesgo de Abandono', '🔴 ALTA (Retención)'
    elif R >= 30 and R <= 90:
        return 'Potencial o Dormido', '🟡 MEDIA (Activación)'
    else:
        return 'Cliente Estándar', '🟢 BAJA'

df_clientes[['Segmento', 'Prioridad de Llamada']] = df_clientes.apply(
    lambda row: pd.Series(get_segmento_y_prioridad(row)), axis=1
)

# 4. Función para Farmacia Más Cercana
def get_farmacia_cercana(lat_c, lon_c):
    min_distancia = float('inf')
    farmacia_cercana = "N/A"
    
    # Calcular distancia Euclidiana simplificada
    for index, row in df_farmacias.iterrows():
        distancia = np.sqrt((lat_c - row['Latitud'])**2 + (lon_c - row['Longitud'])**2)
        
        if distancia < min_distancia:
            min_distancia = distancia
            farmacia_cercana = row['Nombre']
            
    return farmacia_cercana

df_clientes['Farmacia_Cercana'] = df_clientes.apply(
    lambda row: get_farmacia_cercana(row['Latitud_Cliente'], row['Longitud_Cliente']), axis=1
)

# --- VISUALIZACIÓN DE TABLA FRM ---
st.markdown("##### Clientes Prioritarios por FRM y Proximidad")
st.dataframe(
    df_clientes.sort_values(['Prioridad de Llamada', 'Recencia'], ascending=[False, False]),
    column_order=["ID_Cliente", "Prioridad de Llamada", "Segmento", "Recencia", "Farmacia_Cercana", "Monto_Promedio"],
    hide_index=True,
    column_config={
        "Monto_Promedio": st.column_config.NumberColumn(format="Bs. %d")
    }
)
st.success("Acción Proactiva Sugerida: Contactar a clientes con prioridad 🔴 ALTA cerca de su **Farmacia_Cercana** para campañas locales.")

st.markdown("---")

## 3. Visualización de Geolocalización

st.header("3.  Geolocalización de Clientes en Riesgo")
st.markdown("Visualiza dónde se concentran los clientes con prioridad **🔴 ALTA**.")

# 1. Clientes en Riesgo (puntos rojos)
df_mapa_clientes = df_clientes[df_clientes['Prioridad de Llamada'].str.contains('🔴 ALTA')].copy()

# AÑADIDO: Asignar la columna 'Tipo'
df_mapa_clientes['Tipo'] = 'Cliente Riesgo' 

# Renombrar columnas de clientes para Streamlit (lat y lon)
df_mapa_clientes = df_mapa_clientes.rename(columns={
    'Latitud_Cliente': 'lat',
    'Longitud_Cliente': 'lon'
})

# 2. Añadir Farmacias (puntos azules)
# *** ESTA PARTE DEBE SER ASIGNADA CORRECTAMENTE PRIMERO ***
df_farmacias_mapa = df_farmacias.rename(columns={
    'Latitud': 'lat', 
    'Longitud': 'lon'
})
df_farmacias_mapa['Tipo'] = 'Farmacia' # Identificador


# 3. Unir todos los puntos
df_mapa_final = pd.concat([
    # Clientes
    df_mapa_clientes[['lat', 'lon', 'Tipo', 'ID_Cliente', 'Recencia', 'Monto_Promedio']],
    # Farmacias
    df_farmacias_mapa[['lat', 'lon', 'Tipo']].assign(ID_Cliente='Farmacia', Recencia='N/A', Monto_Promedio=0) 
], ignore_index=True)



# 4. Crear Columna para el Color (¡OPCIÓN MÁS ESTABLE CON HEXADECIMAL!)
COLOR_FARMACIA_HEX = '#0000FF'    # azul
COLOR_CLIENTE_HEX = '#FF0000'  # rojo

def get_color_hex(tipo):
    """Devuelve el código Hex basado en el Tipo."""
    if tipo == 'Farmacia':
        return COLOR_FARMACIA_HEX
    return COLOR_CLIENTE_HEX

# Creamos la columna 'color_hex'
df_mapa_final['color_hex'] = df_mapa_final['Tipo'].apply(get_color_hex)


# 5. Crear Columna para el Tamaño (El método de string es más estable)
def get_point_size(tipo):
    """Asigna un tamaño al punto basado en el Tipo."""
    if tipo == 'Farmacia':
        return 400
    return 50

# Creamos la nueva columna 'point_size'
df_mapa_final['point_size'] = df_mapa_final['Tipo'].apply(get_point_size)


# Generar el Mapa de Streamlit
if not df_mapa_final.empty:
    st.map(
        df_mapa_final, 
        latitude='lat', 
        longitude='lon', 
        # Usamos el NOMBRE de la COLUMNA de strings Hex.
        color='color_hex', 
        # Usamos el NOMBRE de la COLUMNA de enteros para el tamaño.
        size='point_size', 
        zoom=11
    )
else:
    st.info("No hay clientes en riesgo (🔴 ALTA) en los datos simulados para mostrar en el mapa.")
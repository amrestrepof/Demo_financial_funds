import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta 

# --- LÓGICA DE DATOS Y MODELO INTEGRADA ---

# FUNCIÓN INTEGRADA 1: GENERACIÓN DE DATOS SIMULADOS PARA RFM
def generate_rfm_data():
    """Simula la base de datos de compras y farmacias, con ubicaciones y fechas."""
    
    # --- Datos de Farmacias (Farmacias fijas) ---
    df_farmacias = pd.DataFrame({
        'Nombre': ['Farmacia Central', 'Farmacia Norte', 'Farmacia Sur'],
        'Latitud': [4.62, 4.70, 4.55],
        'Longitud': [-74.06, -74.12, -74.15]
    })
    
    # --- Datos de Compras (Para calcular RFM) ---
    np.random.seed(42)
    N_CLIENTES = 50
    N_TRANSACCIONES = 300
    
    start_date = datetime.now() - timedelta(days=365)
    
    data_compras = {
        'ID_Cliente': np.random.choice([f'CL{i+1:03d}' for i in range(N_CLIENTES)], size=N_TRANSACCIONES),
        'Fecha_Compra': [start_date + timedelta(days=np.random.randint(1, 365)) for _ in range(N_TRANSACCIONES)],
        'Monto': np.random.randint(15000, 150000, size=N_TRANSACCIONES),
    }
    df_compras = pd.DataFrame(data_compras)
    
    # AJUSTE CLAVE: Forzar clientes a 'EN RIESGO'
    clientes_riesgo = np.random.choice(df_compras['ID_Cliente'].unique(), size=int(N_CLIENTES * 0.2), replace=False)
    fecha_antigua = datetime.now() - timedelta(days=120) 
    
    for cliente in clientes_riesgo:
        df_compras.loc[df_compras['ID_Cliente'] == cliente, 'Fecha_Compra'] = np.nan 
        df_compras = pd.concat([df_compras, pd.DataFrame({
            'ID_Cliente': [cliente], 
            'Fecha_Compra': [fecha_antigua - timedelta(days=np.random.randint(10, 20))], 
            'Monto': [np.random.randint(15000, 50000)],
            'Latitud_Cliente': [4.6],
            'Longitud_Cliente': [-74.0]
        })], ignore_index=True)
        
    df_compras.dropna(subset=['Fecha_Compra'], inplace=True)
    
    # Simular ubicación promedio del cliente
    cliente_ids = df_compras['ID_Cliente'].unique()
    latitudes = 4.6 + np.random.uniform(-0.1, 0.1, len(cliente_ids))
    longitudes = -74.05 + np.random.uniform(-0.15, 0.15, len(cliente_ids))
    ubicacion_map = dict(zip(cliente_ids, zip(latitudes, longitudes)))
    
    df_compras['Latitud_Cliente'] = df_compras['ID_Cliente'].map(lambda x: ubicacion_map.get(x, (4.6, -74.05))[0])
    df_compras['Longitud_Cliente'] = df_compras['ID_Cliente'].map(lambda x: ubicacion_map.get(x, (4.6, -74.05))[1])
    
    return df_compras, df_farmacias

# --- INICIO DEL DEMO STREAMLIT ---

# Cargar solo los datos necesarios para RFM y GEO (función integrada)
try:
    df_compras, df_farmacias = generate_rfm_data()
except Exception as e:
    st.error(f"Error al inicializar los datos para RFM: {e}")
    st.stop()
    
# --- Título Principal ---
st.set_page_config(layout="wide", page_title="Demo Proactivo: Modelo RFM con Geopriorización")
st.title(" Gestión Proactiva: Modelo FRM con Geopriorización (FRM + L)")
st.markdown("---")


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


# 3. Función para Segmentación y Prioridad (Semáforo) Y PRODUCTO SUGERIDO (ALTO VALOR)
def get_segmento_prioridad_y_producto(row):
    R = row['Recencia']
    F = row['Frecuencia']
    
    if R < 30 and F >= 5:
        # Cliente VIP: Ofrecer producto de FIDELIZACIÓN y ALTO MARGEN
        return 'Cliente VIP', '🟡 MEDIA (Fidelización)', 'Protector Solar FPS 50 / Complejo B' 
    elif R > 90 and F < 2:
        # Riesgo ALTO: Ofrecer producto de ALTA RECOMPRA o ESENCIAL, ligado a un DESCUENTO
        return 'En Riesgo de Abandono', '🔴 ALTA (Retención)', 'Multivitamínico Premium (con 15% dcto)'
    elif R >= 30 and R <= 90:
        # Potencial: Ofrecer producto de salud preventiva de valor moderado
        return 'Potencial o Dormido', '🟡 MEDIA (Activación)', 'Suplemento Digestivo o Fibra'
    else:
        # Cliente Estándar: Sin acción proactiva fuerte
        return 'Cliente Estándar', '🟢 BAJA', 'N/A'

df_clientes[['Segmento', 'Prioridad de Llamada', 'Producto_Sugerido']] = df_clientes.apply(
    lambda row: pd.Series(get_segmento_prioridad_y_producto(row)), axis=1
)

# 4. Función para Farmacia Más Cercana (Se mantiene)
def get_farmacia_cercana(lat_c, lon_c):
    min_distancia = float('inf')
    farmacia_cercana = "N/A"
    
    for index, row in df_farmacias.iterrows():
        distancia = np.sqrt((lat_c - row['Latitud'])**2 + (lon_c - row['Longitud'])**2)
        
        if distancia < min_distancia:
            min_distancia = distancia
            farmacia_cercana = row['Nombre']
            
    return farmacia_cercana

df_clientes['Farmacia_Cercana'] = df_clientes.apply(
    lambda row: get_farmacia_cercana(row['Latitud_Cliente'], row['Longitud_Cliente']), axis=1
)

# --- Contenido del Dashboard ---

## 2. Priorización de Clientes (FRM + L)
# -------------------------------------
st.header("2.  Segmentación de Clientes y Prioridad de Contacto")
st.markdown(
    """
    El modelo **FRM (Recencia, Frecuencia, Monto)** identifica el valor del cliente y su **riesgo de abandono**.
    La columna **Prioridad de Llamada** (semáforo) guía al agente hacia la acción más rentable.
    """
)

# --- VISUALIZACIÓN DE TABLA FRM ---
st.markdown("##### Clientes Prioritarios por FRM y Proximidad")

# Aplicar el formato al Monto_Promedio antes de mostrar el DataFrame
df_clientes['Monto_Promedio_COP'] = df_clientes['Monto_Promedio'].apply(lambda x: f"COP {x:,.0f}")

st.dataframe(
    df_clientes.sort_values(['Prioridad de Llamada', 'Recencia'], ascending=[False, False]),
    column_order=["ID_Cliente", "Prioridad de Llamada", "Segmento", "Producto_Sugerido", "Recencia", "Farmacia_Cercana", "Monto_Promedio_COP"],
    hide_index=True,
    column_config={
        "Monto_Promedio_COP": st.column_config.TextColumn("Monto Promedio (COP)")
    }
)
st.success("Acción Proactiva Sugerida: Contactar a clientes con prioridad 🔴 ALTA cerca de su **Farmacia_Cercana** ofreciendo el **Producto Sugerido**.")

st.markdown("---")

## 3. Geolocalización de Clientes en Riesgo
# ---------------------------------------
st.header("3.  Geolocalización de Clientes en Riesgo")
st.markdown(
    """
    Visualizamos la distribución geográfica de los clientes de **Prioridad ALTA (Rojo)** junto a las Farmacias (Azul).
    Esto permite optimizar la logística y el *marketing* de proximidad.
    """
)

# 1. Clientes en Riesgo (puntos rojos)
df_mapa_clientes = df_clientes[df_clientes['Prioridad de Llamada'].str.contains('🔴 ALTA')].copy()

# 2. Preparar Farmacias y Clientes para el Mapa (Mismo código de color y tamaño)
df_mapa_clientes['Tipo'] = 'Cliente Riesgo' 
df_mapa_clientes = df_mapa_clientes.rename(columns={'Latitud_Cliente': 'lat', 'Longitud_Cliente': 'lon'})
df_farmacias_mapa = df_farmacias.rename(columns={'Latitud': 'lat', 'Longitud': 'lon'})
df_farmacias_mapa['Tipo'] = 'Farmacia' 
df_mapa_final = pd.concat([
    df_mapa_clientes[['lat', 'lon', 'Tipo']],
    df_farmacias_mapa[['lat', 'lon', 'Tipo']]
], ignore_index=True)

# 3. Crear Columna para el Color y Tamaño (usando lógica binaria)
COLOR_FARMACIA_HEX = '#0000FF' 
COLOR_CLIENTE_HEX = '#FF0000' 

df_mapa_final['color_hex'] = df_mapa_final['Tipo'].apply(lambda tipo: COLOR_FARMACIA_HEX if tipo == 'Farmacia' else COLOR_CLIENTE_HEX)
df_mapa_final['point_size'] = df_mapa_final['Tipo'].apply(lambda tipo: 400 if tipo == 'Farmacia' else 50)

# Generar el Mapa de Streamlit
if not df_mapa_final.empty:
    st.map(
        df_mapa_final, 
        latitude='lat', 
        longitude='lon', 
        color='color_hex', 
        size='point_size', 
        zoom=11
    )
else:
    st.info("No hay clientes en riesgo (🔴 ALTA) en los datos simulados para mostrar en el mapa.")

st.markdown("---")

## 4. Conclusiones Estratégicas y Rol del Agente (NUEVA SECCIÓN)
# -------------------------------------------------------------
st.header("4.  Conclusiones Estratégicas y Rol del Agente")
st.markdown("---")

col_estrategia, col_rol = st.columns(2)

with col_estrategia:
    st.subheader("Estrategia Impulsada por Data")
    st.markdown(
        """
        - **Mitigación de Riesgo:** El modelo asegura que la inversión (tiempo del agente, descuento) se dirige **solo al 20% de clientes** con la **mayor probabilidad de abandono** (🔴 ALTA). Esto hace que la campaña de retención sea rentable.
        - **Optimización Logística (Geo):** La columna **Farmacia_Cercana** permite a los gerentes optimizar las rutas de entrega o planificar campañas de *marketing* de proximidad altamente segmentadas.
        - **Rentabilidad de la Oferta:** Las sugerencias de productos se enfocan en artículos de **Alto Valor** (ej., Multivitamínicos Premium), maximizando el retorno de la llamada.
        """
    )
    

with col_rol:
    st.subheader("El Agente como Consultor Estratégico")
    st.warning(
        """
        El rol del agente cambia de un simple *vendedor* a un **consultor de retención e inteligencia logística**. 
        
        Su tarea no es vender en frío, sino **ejecutar una estrategia de valor**:
        1. **Justificación:** La llamada está justificada por la prioridad 🔴 ALTA.
        2. **Ofensiva:** El producto ofrecido es la **solución de alto valor** para retener (ej. descuento en el Multivitamínico Premium).
        3. **Cierre Logístico:** Utiliza la **Farmacia_Cercana** para ofrecer conveniencia y asegurar el cierre.
        """
    )
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta 

# --- LÓGICA DE DATOS Y MODELO INTEGRADA ---

# FUNCIÓN INTEGRADA 1: GENERACIÓN DE DATOS SIMULADOS
def generate_data():
    """Simula la base de datos de la Drogueria (Catálogo, Compras, Droguerias)."""
    
    # 1. Catálogo de Productos y Stock (AMPLIADO A 52 PRODUCTOS)
    data_catalogo = {
        'Producto': [
            'Aspirina 500mg', 'Ibuprofeno 400mg', 'Acetaminofén 500mg', 'Dolex Forte', 
            'Amoxicilina 500mg', 'Azitromicina 500mg', 'Omeprazol 20mg', 'Ranitidina 150mg',
            'Loratadina 10mg', 'Cetirizina 10mg', 'Vitamina C', 'Complejo B', 
            'Melatonina 5mg', 'Biotina', 'Pastillas para la Garganta', 
            'Cremas Hidratantes', 'Protector Solar FPS 50', 'Agua Micelar', 
            'Shampoo Anticaída', 'Acondicionador Reparador', 'Termómetro Digital', 
            'Nebulizador Portátil', 'Curitas (Caja)', 'Vendas Elásticas', 
            'Alcohol Antiséptico', 'Agua Oxigenada', 'Gasa Estéril', 'Suero Oral', 
            'Sales Rehidratantes', 'Antiflu Descongestivo', 'Gotas Oftálmicas',
            'Laxante Natural', 'Fibra Soluble', 'Probióticos', 'Tiras Reactivas Glucosa',
            'Tensiómetro Digital', 'Preservativos (Caja)', 'Lubricante Íntimo',
            'Prueba de Embarazo', 'Óxido de Zinc', 'Pañales para Adultos', 
            'Sustituto Lácteo Bebé', 'Leche de Fórmula (Etapa 1)', 
            'Crema para Rozaduras', 'Toallitas Húmedas', 'Tinte para Cabello', 
            'Removedor de Esmalte', 'Jabon Líquido Neutro', 'Enjuague Bucal', 
            'Cepillo Dental Eléctrico', 'Crema Dental Sensible', 'Esponja de Baño'
        ],
        # Simulación de Stock (Stock bajo en 15 productos para activar alertas)
        'Stock': [4, 50, 80, 5, 6, 12, 22, 100, 15, 70, 200, 45, 10, 88, 30, 
                  75, 12, 110, 9, 35, 18, 5, 40, 60, 150, 180, 7, 25, 45, 90, 
                  22, 14, 30, 55, 3, 11, 40, 65, 8, 30, 20, 15, 10, 100, 90, 
                  55, 60, 150, 95, 20, 70, 120],
        # Lógica de Sustitución
        'Sustituto': [
            'Ibuprofeno 400mg', 'Acetaminofén 500mg', 'Dolex Forte', 'Acetaminofén 500mg', 
            'Azitromicina 500mg', np.nan, 'Ranitidina 150mg', 'Omeprazol 20mg',
            'Cetirizina 10mg', 'Loratadina 10mg', 'Complejo B', np.nan,
            np.nan, 'Vitamina C', np.nan, 'Vaselina Pura', 
            'Crema Hidratantes', np.nan, np.nan, 'Shampoo Anticaída', 
            np.nan, np.nan, 'Vendas Elásticas', 'Curitas (Caja)', 
            np.nan, np.nan, np.nan, 'Sales Rehidratantes', 
            'Suero Oral', np.nan, np.nan, 'Fibra Soluble', 
            'Laxante Natural', np.nan, np.nan, np.nan, 
            np.nan, np.nan, np.nan, np.nan, 
            np.nan, np.nan, np.nan, np.nan, 
            np.nan, np.nan, np.nan, np.nan, 
            np.nan, np.nan, np.nan, np.nan
        ],
        # Lógica de Venta Cruzada
        'Complemento': [
            'Pastillas para la Garganta', 'Vitamina C', 'Suero Oral', 'Vitamina C', 
            'Probióticos', 'Suero Oral', 'Complejo B', 'Vitamina C', 
            'Termómetro Digital', 'Vitamina C', 'Protector Solar FPS 50', 'Loratadina 10mg', 
            'Vitamina C', 'Shampoo Anticaída', 'Aspirina 500mg', 'Protector Solar FPS 50', 
            'Cremas Hidratantes', 'Jabon Líquido Neutro', 'Acondicionador Reparador', 'Shampoo Anticaída', 
            'Loratadina 10mg', 'Amoxicilina 500mg', 'Alcohol Antiséptico', 'Curitas (Caja)', 
            'Curitas (Caja)', 'Gasa Estéril', 'Vendas Elásticas', 'Dolex Forte', 
            'Suero Oral', 'Pastillas para la Garganta', 'Loratadina 10mg', 'Vitamina C', 
            'Omeprazol 20mg', 'Amoxicilina 500mg', 'Vitamina C', 'Tiras Reactivas Glucosa', 
            'Lubricante Íntimo', 'Preservativos (Caja)', 'Vitamina C', 'Crema para Rozaduras', 
            'Toallitas Húmedas', 'Crema para Rozaduras', 'Pañales para Adultos', 'Toallitas Húmedas', 
            'Crema para Rozaduras', 'Removedor de Esmalte', 'Tinte para Cabello', 'Enjuague Bucal', 
            'Cepillo Dental Eléctrico', 'Crema Dental Sensible', 'Enjuague Bucal', 'Jabon Líquido Neutro'
        ]
    }
    df_catalogo = pd.DataFrame(data_catalogo)
    
    # 2. Otros DataFrames (Vacíos o Mínimos)
    df_compras = pd.DataFrame()
    df_farmacias = pd.DataFrame({
        'Nombre': ['Drogueria Central', 'Drogueria Norte', 'Drogueria Sur'],
        'Latitud': [4.6, 4.7, 4.5],
        'Longitud': [-74.0, -74.1, -74.2]
    })
    
    return df_catalogo, df_compras, df_farmacias

# FUNCIÓN INTEGRADA 3: GENERACIÓN DE RECOMENDACIONES (Modelo de Cross-Sell/Stock)
def generar_recomendaciones(producto_detectado, df_catalogo):
    """Genera alertas de stock y sugerencias de cross-sell basadas en el catálogo."""
    
    # Buscar el producto, si no lo encuentra, retornar error.
    if producto_detectado not in df_catalogo['Producto'].values:
        return {
            'Alerta_Stock': False,
            'Sustituto_Mensaje': "Producto no encontrado en el catálogo.",
            'Complemento': "N/A"
        }
    
    producto_data = df_catalogo[df_catalogo['Producto'] == producto_detectado].iloc[0]
    
    stock_actual = producto_data['Stock']
    umbral_bajo = 10 # Umbral de alerta
    
    alerta_stock = stock_actual <= umbral_bajo
    sustituto_mensaje = ""
    
    # Lógica de Sustitución
    if alerta_stock:
        sustituto = producto_data['Sustituto']
        if pd.notna(sustituto):
            sustituto_mensaje = f"Stock bajo ({stock_actual} uds). **¡Sustitución Obligatoria!** Ofrecer: **{sustituto}**."
        else:
            sustituto_mensaje = f"Stock bajo ({stock_actual} uds). No hay sustituto directo en el sistema. Venta en riesgo."
    else:
        sustituto_mensaje = "Stock suficiente para la venta."
        
    # Lógica de Venta Cruzada
    complemento = producto_data['Complemento']
    
    return {
        'Alerta_Stock': alerta_stock,
        'Sustituto_Mensaje': sustituto_mensaje,
        'Complemento': complemento
    }


# --- INICIO DEL DEMO STREAMLIT ---

# Cargar datos simulados (usando la función integrada)
try:
    df_catalogo, df_compras, df_farmacias = generate_data()
except Exception as e:
    st.error(f"Error al inicializar los datos: {e}")
    st.stop()
    
# --- Título Principal ---
st.title(" Asistente de Venta Inteligente para Agentes Operaciones Droguerias")
st.markdown("---")
    
# --- Sección Única: Motor de Venta Cruzada y Sustitución ---
st.header("1.  Motor de Recomendación y Gestión de Stock")
st.subheader("Guía Estratégica para Maximizar Rentabilidad y Servicio")

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("#### Entrada del Agente")
    
    # Nuevo: Selectbox para la entrada manual del Agente
    productos_lista = ['Seleccionar Producto'] + sorted(df_catalogo['Producto'].tolist()) # Ordenar la lista para facilitar la búsqueda
    producto_detectado = st.selectbox(
        "Producto Solicitado por el Cliente:",
        options=productos_lista
    )

with col2:
    st.markdown("####  Sugerencias de la IA en Tiempo Real")

    if producto_detectado and producto_detectado != 'Seleccionar Producto':
        st.success(f"**Producto Base Ingresado:** **{producto_detectado}**")

        recomendaciones = generar_recomendaciones(producto_detectado, df_catalogo)

        st.markdown("---")
        
        # 1. Alerta de Stock y Sustitución
        if recomendaciones['Alerta_Stock']:
            st.error(" ¡ACCIÓN INMEDIATA: STOCK BAJO!") 
            st.warning("**Motor de Sustituciones:**")
            st.write(recomendaciones['Sustituto_Mensaje'])
        else:
             st.info(" Stock Suficiente. No requiere sustitución.")

        st.markdown("---")
        
        # 2. Venta Cruzada
        st.info(" **Venta Cruzada (Cross-Sell Sugerido):**")
        st.write(f"Prioridad: Ofrecer **{recomendaciones['Complemento']}** (Aumenta el ticket promedio).")
        
    else:
        st.warning("Agente: Por favor, ingrese el producto solicitado para recibir la guía estratégica.")

st.markdown("---")

st.markdown(
    """
    **Valor Analítico:** La plataforma funciona como un **cerebro de ventas y logística** que da al agente
    las dos acciones más importantes en tiempo real: **resolver la restricción de stock** (evitando perder la venta)
    y **sugerir el producto de mayor valor añadido** (maximizando la rentabilidad).
    """
)
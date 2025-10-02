import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_data():
    # 1. Catálogo de Productos (Definiciones y lógica de Cross-Sell/Sustitución)
    productos = {
        'Aspirina 500mg': {'Cat': 'Analgésico', 'Stock': 50, 'Sustituto': 'Ibuprofeno 400mg', 'Complemento': 'Protector Gástrico'},
        'Ibuprofeno 400mg': {'Cat': 'Analgésico', 'Stock': 10, 'Sustituto': 'Paracetamol 1g', 'Complemento': 'Pomada Muscular'},
        'Vitamina C Effervescent': {'Cat': 'Suplemento', 'Stock': 120, 'Sustituto': 'Multivitamínico Diario', 'Complemento': 'Zinc'},
        'Insulina Glargina': {'Cat': 'Crónico', 'Stock': 5, 'Sustituto': None, 'Complemento': 'Agujas Desechables'},
        # NUEVOS EJEMPLOS:
        'Jarabe para la Tos Seca': {'Cat': 'Resfriado', 'Stock': 80, 'Sustituto': None, 'Complemento': 'Pastillas para la Garganta'},
        'Crema Hidratante Piel Seca': {'Cat': 'Dermatológico', 'Stock': 60, 'Sustituto': 'Crema Corporal Genérica', 'Complemento': 'Protector Solar FPS 50'},
    }
    df_catalogo = pd.DataFrame.from_dict(productos, orient='index').reset_index().rename(columns={'index': 'Producto'})

    # 2. Ubicación de Farmacias (Datos para la Geografía)
    df_farmacias = pd.DataFrame({
        'ID_Farmacia': [1, 2, 3],
        'Nombre': ['Central', 'Norte', 'Sur'],
        # Coordenadas simuladas (ej. Medellín, Colombia)
        'Latitud': [6.2442, 6.2600, 6.2200],  
        'Longitud': [-75.5812, -75.5750, -75.5900]
    })

    # 3. Historial de Compras (Datos para FRM y Geolocalización)
    clientes = [f'C{i:03d}' for i in range(100)]
    
    # *** CÓDIGO FALTANTE AÑADIDO: Generación de Fechas ***
    # Simula 500 compras distribuidas en el último año
    fechas = [datetime.now() - timedelta(days=np.random.randint(1, 365)) for _ in range(500)]
    
    df_compras = pd.DataFrame({
        'ID_Cliente': np.random.choice(clientes, 500),
        'Fecha_Compra': fechas,
        'Monto': np.random.randint(5, 150, 500) * 1000,
        'Producto_Principal': np.random.choice(df_catalogo['Producto'], 500),
        # SIMULACIÓN DE LA UBICACIÓN DEL CLIENTE
        # loc (mean): centro de coordenadas simuladas
        # scale (std dev): dispersión de clientes
        'Latitud_Cliente': np.random.normal(loc=6.24, scale=0.03, size=500), 
        'Longitud_Cliente': np.random.normal(loc=-75.58, scale=0.03, size=500),
    })

    # 4. Devolución de los DataFrames
    return df_catalogo, df_compras, df_farmacias

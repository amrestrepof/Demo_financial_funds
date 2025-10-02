import pandas as pd

def detectar_producto(texto_llamada):
    texto_llamada = texto_llamada.lower()
    
    # Detección por nombre específico de producto y síntomas
    productos_simulados = {
        'aspirina 500mg': 'Aspirina 500mg', 
        'ibuprofeno 400mg': 'Ibuprofeno 400mg', 
        'insulina glargina': 'Insulina Glargina',
        'vitamina c': 'Vitamina C Effervescent',
        'jarabe': 'Jarabe para la Tos Seca',
        'tos seca': 'Jarabe para la Tos Seca', 
        'crema hidratante': 'Crema Hidratante Piel Seca',
    }
    
    for keyword, product_name in productos_simulados.items():
        if keyword in texto_llamada:
            return product_name
    return None

def generar_recomendaciones(producto_base, df_catalogo):
    if not producto_base:
        # Usamos la clave esperada por app.py
        return {'Sustituto_Mensaje': 'N/A', 'Complemento': 'N/A', 'Alerta_Stock': False}

    prod_info = df_catalogo[df_catalogo['Producto'] == producto_base].iloc[0]
    stock_actual = prod_info['Stock']
    
    # 1. Lógica de Alerta de Stock
    alerta_stock = stock_actual < 15 # Umbral bajo
    
    # 2. Lógica de Sustitución y Generación del Mensaje Único
    sustituto = prod_info['Sustituto']
    sustituto_mensaje = "Producto en stock o sin sustituto idóneo."
    
    if alerta_stock:
        # El producto está bajo en stock, generamos el mensaje de alerta/sustitución
        if sustituto:
            sustituto_mensaje = f"⚠️ ¡STOCK BAJO ({stock_actual} uds)! Sugiera el sustituto: **{sustituto}**."
        else:
            sustituto_mensaje = f"⚠️ ¡STOCK BAJO ({stock_actual} uds)! No hay sustituto directo. Pedir confirmación a bodega."
            
    # 3. Lógica de Cross-Sell (Complemento)
    complemento = prod_info['Complemento']

    # DEVOLUCIÓN AJUSTADA con la clave 'Sustituto_Mensaje'
    return {
        'Sustituto_Mensaje': sustituto_mensaje,
        'Complemento': complemento,
        'Alerta_Stock': alerta_stock
    }
import pandas as pd
from faker import Faker
import random
from datetime import datetime

# Inicializar Faker
fake = Faker('es_ES')

def generar_dataset_bancario(n_clientes=1000):
    """Genera un dataset realista para un demo de churn y segmentación bancaria."""
    data = []
    
    for i in range(n_clientes):
        # --- Características del Cliente ---
        edad = random.randint(18, 75)
        antiguedad_meses = random.randint(1, 120)
        num_productos = random.randint(1, 6) # Cuentas, tarjetas, préstamos, etc.
        tiene_tarjeta_credito = random.choice([0, 1])
        usa_app_movil = random.choice([0, 1])
        
        # --- Comportamiento Transaccional ---
        saldo_cuenta = round(random.uniform(0, 150000), 2)
        score_crediticio = random.randint(300, 850)
        num_transacciones_mes = random.randint(0, 100)
        
        # --- Interacciones con el Banco ---
        quejas_ultimo_trimestre = random.randint(0, 4)
        
        # --- Lógica de Churn (Variable Objetivo) ---
        # Creamos una probabilidad de churn basada en reglas de negocio lógicas
        prob_churn = 0.05 # Base
        if antiguedad_meses < 12: prob_churn += 0.10
        if num_productos == 1: prob_churn += 0.15
        if quejas_ultimo_trimestre > 1: prob_churn += 0.20 * quejas_ultimo_trimestre
        if saldo_cuenta < 1000: prob_churn += 0.05
        if score_crediticio < 500: prob_churn += 0.10
        if usa_app_movil == 0: prob_churn += 0.05

        # El churn es más probable si la probabilidad es alta
        churn = 1 if random.random() < prob_churn else 0
        
        data.append({
            "id_cliente": 10000 + i,
            "edad": edad,
            "antiguedad_meses": antiguedad_meses,
            "num_productos": num_productos,
            "tiene_tarjeta_credito": tiene_tarjeta_credito,
            "usa_app_movil": usa_app_movil,
            "saldo_cuenta": saldo_cuenta,
            "score_crediticio": score_crediticio,
            "num_transacciones_mes": num_transacciones_mes,
            "quejas_ultimo_trimestre": quejas_ultimo_trimestre,
            "churn": churn
        })
        
    return pd.DataFrame(data)

if __name__ == '__main__':
    df_banco = generar_dataset_bancario(2000)
    df_banco.to_csv('clientes_banco.csv', index=False)
    print("Archivo 'clientes_banco.csv' generado con éxito.")
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
import plotly.express as px
import matplotlib.pyplot as plt

st.set_page_config(page_title="Demo Telco: Ventas & Up-selling", layout="wide")

st.title("Demo IA Telco — Conversión de Ventas & Up-selling / Cross-selling")
st.markdown("""
Este demo muestra cómo técnicas de **Machine Learning** pueden ayudar a optimizar la gestión de ventas y maximizar la conversión en campañas comerciales de Telco.
""")

# --- 1. SIMULACIÓN DE DATOS ---
with st.expander("1️⃣ ¿Cómo se crean los datos del demo? (Simulación realista)"):
    st.info(
        "Simulamos clientes contactados por el call center, con variables como: número de llamadas, productos ofrecidos, perfil demográfico, historial de compras y resultado de las últimas llamadas."
    )

np.random.seed(123)
N = 1000
data = pd.DataFrame({
    'Edad': np.random.randint(18, 70, N),
    'Sexo': np.random.choice(['M', 'F'], N),
    'Segmento': np.random.choice(['Joven', 'Adulto', 'Senior'], N, p=[0.3,0.5,0.2]),
    'Contactos': np.random.poisson(2, N)+1,
    'Productos_ofrecidos': np.random.choice(['Internet', 'TV', 'Movil', 'Paquete Triple'], N),
    'Historial_compras': np.random.randint(0, 5, N),
    'Ultima_llamada': np.random.choice(['Exitosa', 'No exitosa', 'No contesta'], N, p=[0.4, 0.4, 0.2]),
    'Agente': np.random.choice([f'Agente {i+1}' for i in range(10)], N)
})

# Simula probabilidad de compra (target)
def simular_compra(row):
    score = 0
    score += 0.4 if row['Ultima_llamada']=='Exitosa' else -0.2
    score += 0.3 if row['Productos_ofrecidos']=='Paquete Triple' else 0
    score += 0.2*row['Historial_compras']
    score += 0.2 if row['Segmento']=='Adulto' else 0
    score += 0.1 if row['Contactos']>2 else 0
    return 1 if np.random.rand()<1/(1+np.exp(-score)) else 0

data['Compra'] = data.apply(simular_compra, axis=1)

# --- 2. PREPROCESAMIENTO ---
with st.expander("2️⃣ ¿Cómo se preparan los datos?"):
    st.info(
        "Las variables categóricas se convierten en variables numéricas (one-hot encoding). Se separan datos de entrenamiento y prueba para evaluar la capacidad predictiva."
    )

data_enc = pd.get_dummies(data, columns=['Sexo','Segmento','Productos_ofrecidos','Ultima_llamada','Agente'], drop_first=True)
X = data_enc.drop('Compra', axis=1)
y = data_enc['Compra']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

# --- 3. ENTRENAMIENTO DEL MODELO ---
with st.expander("3️⃣ ¿Qué modelo se usa y para qué?"):
    st.info(
        "Se utiliza una **Regresión Logística** para predecir la probabilidad de compra por cliente tras la campaña. Este modelo es transparente y fácil de interpretar para negocio."
    )

clf = LogisticRegression(max_iter=500)
clf.fit(X_train, y_train)
y_pred_proba = clf.predict_proba(X_test)[:,1]
roc = roc_auc_score(y_test, y_pred_proba)
cm = confusion_matrix(y_test, (y_pred_proba>0.5).astype(int))

# --- 4. EVALUACIÓN DEL MODELO ---
with st.expander("4️⃣ ¿Qué tan bueno es el modelo para predecir compras?"):
    st.info(
        "- El **AUC** mide qué tan bien el modelo predice qué clientes comprarán tras el contacto. "
        "La **matriz de confusión** muestra aciertos y errores en la predicción."
    )
st.subheader(f"Rendimiento del Modelo: AUC = {roc:.2f}")
st.write("Matriz de confusión (umbral 0.5):")
st.write(pd.DataFrame(cm, index=["No Compra", "Compra"], columns=["Pred No Compra", "Pred Compra"]))

st.markdown(f"""
**¿Cómo interpretar estos resultados?**

- Un **AUC alto** indica que el modelo distingue muy bien entre quienes comprarán y quienes no.
- La matriz de confusión te dice cuántos clientes predijimos correctamente, y cuántos se nos escapan (falsos positivos/negativos).
""")

# --- 5. IMPORTANCIA DE VARIABLES ---
with st.expander("5️⃣ ¿Qué variables impactan más en la compra?"):
    st.info(
        "Aquí se observa qué variables tienen mayor influencia en la probabilidad de compra (por ejemplo, resultado de la última llamada, tipo de producto ofrecido, historial de compras, etc.)."
    )

importancias = pd.Series(abs(clf.coef_[0]), index=X.columns).sort_values(ascending=False)[:10]
fig1 = px.bar(importancias, x=importancias.index, y=importancias.values, title="Top 10 variables más influyentes")
st.plotly_chart(fig1, use_container_width=True)

st.markdown("""
**¿Qué nos dice este análisis?**

- Las variables más influyentes te permiten ajustar la estrategia (por ejemplo, ofrecer paquetes triples a segmentos adultos que ya han comprado antes aumenta la conversión).
""")

# --- 6. PROBABILIDAD DE COMPRA POR CLIENTE ---
with st.expander("6️⃣ ¿Quiénes tienen mayor probabilidad de comprar?"):
    st.info(
        "Se muestra una tabla con los clientes de mayor probabilidad de compra según el modelo. "
        "Puedes priorizar a estos clientes para una acción más personalizada o cross-selling."
    )

data_test = X_test.copy()
data_test['Prob_Compra'] = y_pred_proba
data_test['Compra_real'] = y_test.values

top_n = st.slider("¿Cuántos clientes con mayor probabilidad de compra deseas ver?", 5, 50, 10)
top_buyers = data_test.sort_values("Prob_Compra", ascending=False).head(top_n)
st.dataframe(top_buyers[['Prob_Compra'] + [col for col in top_buyers.columns if 'Edad' in col or 'Contactos' in col or 'Historial_compras' in col]])

st.markdown("""
**¿Cómo usar esta tabla?**

- Prioriza a los clientes con mayor probabilidad de compra para campañas de cierre rápido, venta cruzada o referidos.
""")

# --- 7. SIMULACIÓN DE RECOMENDACIÓN DE PRODUCTO ---
with st.expander("7️⃣ ¿Qué producto es mejor ofrecerle a cada cliente?"):
    st.info(
        "Basado en el perfil y comportamiento, simulamos una recomendación simple del producto más adecuado para cada cliente, maximizando la tasa de conversión y up-selling."
    )

# Asignación de producto recomendado según perfil y última compra
def recomendar_producto(row):
    if row['Historial_compras'] == 0:
        return "Internet"
    elif row['Historial_compras'] == 1 and row['Edad'] < 35:
        return "Movil"
    elif row['Historial_compras'] > 1:
        return "Paquete Triple"
    else:
        return "TV"
data_test['Producto_Recomendado'] = data_test.apply(recomendar_producto, axis=1)

st.dataframe(data_test[['Prob_Compra', 'Edad', 'Historial_compras', 'Producto_Recomendado']].sort_values("Prob_Compra", ascending=False).head(10))

st.markdown("""
**¿Cómo se puede usar esto?**
- Personaliza la oferta por cliente según su historial, edad y preferencia.
- Aumenta la probabilidad de venta ofreciendo el producto correcto en el momento adecuado.
""")

# --- 8. RANKING DE AGENTES ---
with st.expander("8️⃣ ¿Qué agentes generan más ventas?"):
    st.info(
        "El ranking de agentes permite detectar buenas prácticas y agentes de alto desempeño, útiles para capacitación o incentivos."
    )

# Ranking real de agentes (según predicción)
agentes_cols = [c for c in data_test.columns if c.startswith("Agente_")]
ranking = pd.DataFrame({
    "Agente": [c.replace("Agente_","Agente ") for c in agentes_cols],
    "Score_Ventas": [data_test[c].dot(data_test['Prob_Compra']) for c in agentes_cols]
}).sort_values("Score_Ventas", ascending=False)
fig2 = px.bar(ranking, x="Agente", y="Score_Ventas", title="Ranking de Agentes según Score de Ventas")
st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
**¿Por qué es útil?**
- Detecta a los agentes más efectivos, permitiendo replicar buenas prácticas y diseñar incentivos justos.
""")

# --- 9. BONUS: ANÁLISIS DE MOTIVOS DE COMPRA (Embeddings NLP) ---
with st.expander("9️⃣ ¿Qué motiva la compra? (Deep Learning NLP Simulado)"):
    st.info(
        "Aquí simulamos la extracción automática de motivos de compra usando análisis de texto y Deep Learning (por ejemplo, con BERT o embeddings)."
    )
motivos = [
    "Precio promocional", "Mejor velocidad", "Atención personalizada", "Oferta exclusiva",
    "Recomendación de amigo", "Promoción solo por hoy", "Renovación de contrato", "Beneficios para familia"
]
data_test['Motivo_Compra'] = np.random.choice(motivos, data_test.shape[0])

fig3 = px.bar(data_test['Motivo_Compra'].value_counts(), 
              x=data_test['Motivo_Compra'].value_counts().index, 
              y=data_test['Motivo_Compra'].value_counts().values, 
              title="Motivos más frecuentes de compra detectados en transcripciones (simulación NLP)")
st.plotly_chart(fig3, use_container_width=True)

st.markdown("""
**¿Por qué agregar análisis de texto con IA?**
- Descubrir los verdaderos motivos de compra permite ajustar guiones, entrenar agentes y lanzar ofertas mucho más efectivas y personalizadas.
""")

st.success("¡Listo! Este demo muestra cómo la IA puede transformar el área comercial y de ventas en Telco, desde la predicción hasta la recomendación y el análisis de desempeño.")


##streamlit run app_ventas.py
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
import plotly.express as px
import matplotlib.pyplot as plt

st.set_page_config(page_title="Demo Telco: Predicción de Churn", layout="wide")

st.title(" Demo IA Telco — Predicción de Fuga de Clientes (Churn)")
st.markdown("""
Este demo interactivo simula cómo un algoritmo de **Machine Learning** ayuda a predecir la fuga de clientes en una empresa Telco, permitiendo tomar acciones proactivas para retenerlos.
""")

# --- 1. SIMULACIÓN DE DATOS ---
with st.expander("1️ ¿Cómo se crean los datos del demo? (Simulación realista)"):
    st.info(
        "Se genera un conjunto de datos **Simulados** que representa clientes de Telco, "
        "con variables como edad, antigüedad, reclamos, pagos atrasados, consumo de datos, tipo de plan y satisfacción. "
        "El objetivo ('Churn') indica si el cliente se fue o no, en función de su comportamiento."
    )

np.random.seed(42)
N = 1000
data = pd.DataFrame({
    'Edad': np.random.randint(18, 80, N),
    'Antigüedad_meses': np.random.randint(1, 72, N),
    'Reclamos_ult_6m': np.random.poisson(1, N),
    'Pagos_atrasados': np.random.binomial(4, 0.2, N),
    'Consumo_MB': np.random.normal(5000, 2000, N).clip(0),
    'Tipo_plan': np.random.choice(['Prepago', 'Pospago'], N, p=[0.3, 0.7]),
    'Satisfaccion': np.random.randint(1, 11, N)
})
data['Churn'] = (
    (data['Reclamos_ult_6m'] > 2) |
    (data['Pagos_atrasados'] > 1) |
    (data['Satisfaccion'] < 5)
).astype(int)
data['Churn'] = np.where(np.random.rand(N) < 0.07, 1-data['Churn'], data['Churn'])

# --- 2. PREPROCESAMIENTO ---
with st.expander("2️ ¿Cómo se preparan los datos? (Preprocesamiento)"):
    st.info(
        "Las variables de tipo texto, como 'Tipo de plan', se convierten a variables numéricas "
        "(one-hot encoding). Se separan los datos en conjuntos de entrenamiento y prueba, "
        "para evaluar el modelo de forma objetiva."
    )

data_enc = pd.get_dummies(data, columns=['Tipo_plan'], drop_first=True)
X = data_enc.drop('Churn', axis=1)
y = data_enc['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

# --- 3. ENTRENAMIENTO DEL MODELO ---
with st.expander("3️ ¿Qué modelo se usa y cómo aprende?"):
    st.info(
        "Se utiliza un **Random Forest Classifier**, un modelo de Machine Learning que combina muchos árboles de decisión "
        "para clasificar clientes según su probabilidad de fuga. El modelo aprende patrones a partir de los datos históricos."
    )

clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)
y_pred_proba = clf.predict_proba(X_test)[:,1]
roc = roc_auc_score(y_test, y_pred_proba)
cm = confusion_matrix(y_test, (y_pred_proba>0.5).astype(int))

# --- 4. EVALUACIÓN DEL MODELO ---
with st.expander("4️ ¿Cómo sabemos si el modelo es bueno? (Evaluación)"):
    st.info(
        "- El **AUC** mide la capacidad del modelo para distinguir clientes que se van vs. los que se quedan. "
        "Un valor cercano a 1 es muy bueno. \n"
        "- La **matriz de confusión** muestra cuántos casos fueron clasificados correctamente o no."
    )
st.subheader(f"Rendimiento del Modelo: AUC = {roc:.2f}")
st.write("Matriz de confusión (umbral 0.5):")
st.write(pd.DataFrame(cm, index=["No Churn", "Churn"], columns=["Pred No Churn", "Pred Churn"]))

st.markdown("""
**Interpretación de los resultados del modelo:**

- **AUC = {:.2f}:** Esto significa que el modelo es muy bueno diferenciando entre clientes que se van (churn) y los que se quedan.
- **Matriz de confusión:**
    - **No Churn/Pred No Churn:** Clientes fieles que el modelo identificó correctamente como no fugados.
    - **No Churn/Pred Churn:** Falsos positivos; clientes fieles que el modelo consideró en riesgo (podrían recibir ofertas innecesarias).
    - **Churn/Pred No Churn:** Falsos negativos; clientes fugados que el modelo no logró detectar (oportunidad de mejorar retención).
    - **Churn/Pred Churn:** Churns detectados correctamente, estos clientes pueden ser contactados de forma proactiva para evitar la fuga.

**¿Por qué es útil?**
Permite priorizar esfuerzos: los equipos de retención pueden enfocar campañas y recursos en los clientes realmente en riesgo, maximizando la efectividad y el retorno de la inversión.
""".format(roc))

# --- 5. IMPORTANCIA DE VARIABLES ---
with st.expander("5️ ¿Qué variables influyen más en la fuga de clientes?"):
    st.info(
        "El modelo identifica qué variables son **más importantes** para predecir el churn, ayudando a "
        "priorizar dónde intervenir (por ejemplo: satisfacción, reclamos, pagos atrasados, etc.)."
    )

importancias = pd.DataFrame({
    "Variable": X.columns,
    "Importancia": clf.feature_importances_
}).sort_values("Importancia", ascending=False)
fig1 = px.bar(importancias, x="Variable", y="Importancia", title="Importancia de Variables para el Modelo")
st.plotly_chart(fig1, use_container_width=True)

st.markdown("""
**¿Qué nos dice esta gráfica?**

- La variable más influyente para anticipar la fuga de clientes es la **Satisfacción**: los clientes menos satisfechos tienen mayor probabilidad de irse.
- También son muy relevantes los **pagos atrasados** y la cantidad de **reclamos**.
- Otras variables como el consumo, la antigüedad y el tipo de plan tienen menor peso, pero pueden ser útiles en segmentos específicos.

**¿Por qué es útil?**
Saber qué factores impactan la fuga permite diseñar acciones focalizadas (por ejemplo, mejorar la atención a quienes reportan más reclamos o dar incentivos a quienes bajan su satisfacción).
""")

# --- 6. CLIENTES EN RIESGO ---
with st.expander("6️ ¿Quiénes son los clientes en mayor riesgo?"):
    st.info(
        "Se identifican los clientes con mayor probabilidad de fuga, para priorizar acciones de retención "
        "personalizadas y evitar pérdidas."
    )

data_test = X_test.copy()
data_test['Prob_Churn'] = y_pred_proba
data_test['Churn_real'] = y_test.values

top_n = st.slider("¿Cuántos clientes en riesgo deseas ver?", 5, 50, 10)
top_risk = data_test.sort_values("Prob_Churn", ascending=False).head(top_n)
st.dataframe(top_risk[['Prob_Churn'] + [col for col in top_risk.columns if 'Tipo_plan' in col or col in ['Edad', 'Antigüedad_meses', 'Reclamos_ult_6m', 'Pagos_atrasados', 'Consumo_MB', 'Satisfaccion', 'Churn_real']]])

st.markdown("""
**¿Qué hacer con estos clientes?**

Esta tabla muestra a los clientes con mayor riesgo de fuga. Se recomienda:
- Contactarlos con ofertas personalizadas o soluciones a sus problemas.
- Priorizar estos casos para la acción inmediata del equipo de retención, evitando así la pérdida de ingresos.
""")

# --- 7. DISTRIBUCIÓN DE RIESGO ---
with st.expander("7️ ¿Cómo se distribuye el riesgo de fuga en la base de clientes?"):
    st.info(
        "Esta gráfica muestra la distribución de probabilidades de fuga para todos los clientes, "
        "ayudando a visualizar cuántos están en zona crítica y dónde enfocar recursos."
    )

fig2 = px.histogram(data_test, x="Prob_Churn", nbins=30, color=(data_test.Prob_Churn > 0.5).astype(str),
                    labels={"color": "¿Riesgo alto?"}, title="Distribución de probabilidad de Fuga")
st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
**¿Qué vemos aquí?**

- La gráfica muestra la distribución de la probabilidad de fuga para todos los clientes.
- Aquellos con probabilidad superior a 0.5 están en "zona de riesgo alto" y deben ser priorizados.
- Esto ayuda a dimensionar el tamaño del reto y a planificar cuántos recursos necesitas para intervenir a tiempo.

**¿Por qué es útil?**
Permite definir el tamaño de campañas, estimar impacto potencial y ajustar la estrategia de retención.
""")

# --- 8. MAPA DE CALOR DE SEGMENTOS ---
with st.expander("8️ ¿En qué segmentos hay más riesgo de fuga? (Mapa de calor)"):
    st.info(
        "El mapa de calor permite ver combinaciones de variables críticas (como reclamos y satisfacción) "
        "donde se concentra el mayor riesgo de churn. Así se detectan segmentos a priorizar."
    )

pivot = pd.crosstab(data_test['Reclamos_ult_6m'], data_test['Satisfaccion'], 
                    values=data_test['Prob_Churn'], aggfunc='mean').fillna(0)
fig3, ax = plt.subplots(figsize=(8,4))
im = ax.imshow(pivot, aspect='auto', cmap='coolwarm')
plt.colorbar(im, ax=ax)
ax.set_xlabel("Satisfacción")
ax.set_ylabel("Reclamos últimos 6 meses")
ax.set_title("Mapa de calor: Probabilidad de Fuga")
st.pyplot(fig3)

st.markdown("""
**¿Cómo usar este mapa?**

- Aquí se observa qué combinaciones de **reclamos** y **satisfacción** concentran el mayor riesgo de fuga.
- Por ejemplo, clientes con muchos reclamos y baja satisfacción tienen el riesgo más alto.
- Permite diseñar campañas específicas para esos segmentos (mejorar procesos, hacer seguimiento especial, etc).

**¿Por qué es útil?**
Enfoca los esfuerzos y recursos en los segmentos críticos, logrando mayor eficiencia y resultados.
""")

# --- 9. ANÁLISIS DE CLIENTES QUE SE QUEDAN (NO CHURN) ---

with st.expander("9️ ¿Cómo son los clientes fieles y cómo aprovecharlos?"):
    st.info(
        "Además de identificar el riesgo, es clave conocer el perfil de los clientes fieles. "
        "Esto permite potenciar la fidelización y detectar oportunidades de venta cruzada."
    )

# Calcula promedios por grupo (No Churn vs Churn)
grupo_stats = data_test.copy()
grupo_stats['Churn_real_str'] = grupo_stats['Churn_real'].replace({0:'No Churn', 1:'Churn'})
resumen = grupo_stats.groupby('Churn_real_str')[['Edad', 'Antigüedad_meses', 'Reclamos_ult_6m', 'Pagos_atrasados', 'Consumo_MB', 'Satisfaccion']].mean().T

st.dataframe(resumen.style.highlight_max(axis=1, color='lightgreen').highlight_min(axis=1, color='#FFDDDD'))

st.markdown("""
**¿Qué nos muestra la tabla?**

- Los **clientes fieles** suelen tener mayor satisfacción, menor cantidad de reclamos y menos pagos atrasados.
- Tienen, en promedio, mayor antigüedad y consumen más servicios (¡clientes rentables!).
- Estos perfiles son ideales para campañas de venta cruzada, programas de referidos o incentivos de lealtad.

**Recomendación práctica:**
No descuides a los clientes fieles. Premia su lealtad, ofréceles productos premium o recompensas exclusivas, y conviértelos en embajadores de la marca.
""")


st.success("¡Listo! Este demo muestra paso a paso cómo la IA ayuda a anticipar la fuga de clientes y a tomar decisiones inteligentes en Telco. Puedes personalizarlo para otras áreas del negocio.")


#streamlit run "Demo de Predicción de Churn (Retención).py"
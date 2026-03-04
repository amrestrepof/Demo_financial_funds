import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split
import plotly.express as px
import matplotlib.pyplot as plt

st.set_page_config(page_title="Demo Telco: Cartera - Pago o Mora", layout="wide")

st.title(" Demo IA Telco — Predicción de Pago o Mora (Cartera)")
st.markdown("""
Este demo muestra cómo técnicas de **Machine Learning avanzado (XGBoost)** ayudan a anticipar el riesgo de mora, clasificar clientes según probabilidad de pago y priorizar la gestión de cobranza.
""")

# --- 1. SIMULACIÓN DE DATOS (CON MESES) ---
with st.expander("1️ ¿Cómo se crean los datos del demo? (Simulación realista)"):
    st.info(
        "Simulamos una base de clientes con variables clave de cartera: días de mora, llamadas de cobranza, monto pendiente, promesas de pago anteriores y score de crédito, generados para varios meses para analizar tendencias."
    )

np.random.seed(321)
N = 1000
# Datos base
data_base = pd.DataFrame({
    'ClienteID': np.arange(N),
    'Dias_mora': np.random.randint(0, 60, N),
    'Llamadas_cobranza': np.random.poisson(1, N) + 1,
    'Monto_pendiente': np.random.exponential(300, N).astype(int) + 50,
    'Promesas_incumplidas': np.random.binomial(3, 0.3, N),
    'Score_credito': np.random.normal(650, 60, N).clip(450, 850)
})

def simula_pago(row):
    score = 0
    score -= 0.03 * row['Dias_mora']
    score -= 0.25 * row['Llamadas_cobranza']
    score -= 0.002 * row['Monto_pendiente']
    score -= 0.35 * row['Promesas_incumplidas']
    score += 0.004 * row['Score_credito']
    score += np.random.normal(0, 0.5)
    return 1 if 1/(1+np.exp(-score)) > 0.5 else 0

# Simula historial mensual para animación
meses = ['2024-01', '2024-02', '2024-03', '2024-04', '2024-05', '2024-06']
data_full = pd.DataFrame()

for mes in meses:
    dtmp = data_base.copy()
    dtmp['Mes'] = mes
    # Varía un poco las variables mes a mes para simular dinámica real
    dtmp['Dias_mora'] = (dtmp['Dias_mora'] + np.random.randint(-5, 6, len(dtmp))).clip(0)
    dtmp['Monto_pendiente'] = (dtmp['Monto_pendiente'] * np.random.uniform(0.95, 1.05, len(dtmp))).astype(int)
    dtmp['Pago'] = dtmp.apply(simula_pago, axis=1)
    data_full = pd.concat([data_full, dtmp], ignore_index=True)

# Selecciona un mes (el último) para el análisis detallado del modelo
data = data_full[data_full['Mes'] == meses[-1]].drop(columns=['Mes', 'ClienteID'])

# --- 2. PREPROCESAMIENTO ---
with st.expander("2️ ¿Cómo se preparan los datos?"):
    st.info(
        "Los datos se dividen en entrenamiento y prueba. Se usan para ajustar y validar el modelo XGBoost, que es altamente eficaz para clasificación de riesgo."
    )

X = data.drop('Pago', axis=1)
y = data['Pago']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.25, random_state=42)

# --- 3. ENTRENAMIENTO DEL MODELO (XGBoost) ---
with st.expander("3️ ¿Qué modelo se usa y por qué?"):
    st.info(
        "Se utiliza **XGBoost**, un modelo avanzado que combina muchos árboles de decisión para predecir la probabilidad de pago. Es robusto, preciso y muy usado en banca y telecomunicaciones."
    )

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
params = {'objective': 'binary:logistic', 'eval_metric':'auc', 'seed':42, 'verbosity':0}
bst = xgb.train(params, dtrain, num_boost_round=60)
y_pred_proba = bst.predict(dtest)
roc = roc_auc_score(y_test, y_pred_proba)
cm = confusion_matrix(y_test, (y_pred_proba>0.5).astype(int))

# --- 4. CURVA ROC/AUC Y MATRIZ DE CONFUSIÓN ---
with st.expander("4️ ¿Qué tan bueno es el modelo? (Curva ROC/AUC)"):
    st.info(
        "La **curva ROC** muestra la capacidad del modelo para distinguir entre quienes pagarán y quienes no. El **AUC** cercano a 1 indica excelente discriminación."
    )

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f'AUC = {roc:.2f}')
ax.plot([0,1],[0,1],'--',color='gray')
ax.set_xlabel('Falsos positivos')
ax.set_ylabel('Verdaderos positivos')
ax.set_title('Curva ROC')
ax.legend()
st.pyplot(fig)

st.subheader(f"AUC del modelo: {roc:.2f}")
st.write("Matriz de confusión (umbral 0.5):")
st.write(pd.DataFrame(cm, index=["No Pago", "Pago"], columns=["Pred No Pago", "Pred Pago"]))

st.markdown("""
**¿Cómo interpretar estos resultados?**

- Un **AUC alto** significa que el modelo predice muy bien qué clientes pagarán y cuáles no.
- La matriz de confusión permite ver aciertos y errores. Ideal para ajustar campañas de cobranza.
""")

# --- 5. IMPORTANCIA DE VARIABLES ---
with st.expander("5️ ¿Qué variables son más importantes para predecir el pago?"):
    st.info(
        "El modelo identifica los factores de mayor impacto en la probabilidad de pago (por ejemplo: score de crédito, días de mora, promesas incumplidas)."
    )

importancias = pd.Series(bst.get_score(importance_type='gain')).sort_values(ascending=False)
fig1 = px.bar(importancias, x=importancias.index, y=importancias.values, title="Importancia de variables")
st.plotly_chart(fig1, use_container_width=True)

st.markdown("""
**¿Por qué es útil esto?**
- Permite enfocar esfuerzos de cobranza en los clientes de mayor riesgo según las variables clave.
""")

# ----------- G1. CASCADA DE IMPACTO DE VARIABLES (SIMPLE EXPLICABILIDAD) -----------
with st.expander(" ¿Por qué el modelo clasifica a un cliente como 'alto riesgo'? (Cascada de variables)"):
    st.info("Muestra cómo cada variable afecta la predicción final para un cliente seleccionado.")
    data_test = X_test.copy()
    data_test['Prob_Pago'] = y_pred_proba
    data_test['Pago_real'] = y_test.values
    idx = st.slider("Elige el cliente a analizar (por índice)", 0, len(data_test)-1, 0)
    cliente = data_test.iloc[idx]
    base = 0.5  # Probabilidad base
    features = ['Dias_mora', 'Llamadas_cobranza', 'Monto_pendiente', 'Promesas_incumplidas', 'Score_credito']
    impactos = [
        -0.03 * cliente['Dias_mora'],
        -0.25 * cliente['Llamadas_cobranza'],
        -0.002 * cliente['Monto_pendiente'],
        -0.35 * cliente['Promesas_incumplidas'],
        0.004 * cliente['Score_credito']
    ]
    contribs = pd.DataFrame({
        'Variable': features,
        'Impacto': impactos
    })
    contribs['Acumulado'] = base + contribs['Impacto'].cumsum()
    figw = px.bar(contribs, x='Variable', y='Impacto', title="Impacto de variables en el score de riesgo (Cliente seleccionado)")
    st.plotly_chart(figw, use_container_width=True)
    st.markdown(f"""
    **Interpretación:**  
    - Las barras muestran cómo cada variable aumenta/disminuye la probabilidad de pago.
    - El score final para este cliente es: **{cliente['Prob_Pago']:.2f}**
    """)

# ----------- G2. DISPERSIÓN SCORE VS DÍAS DE MORA (COLOR POR RIESGO) -----------
with st.expander(" ¿Cómo se agrupan los clientes según score y días de mora? (Dispersión interactiva)"):
    st.info("Visualiza los clientes en dos dimensiones críticas: Score de crédito y días de mora, coloreados por riesgo.")
    # Clasificación por riesgo para graficar
    condiciones = [
        (data_test['Prob_Pago'] > 0.8),
        (data_test['Prob_Pago'] > 0.5),
        (data_test['Prob_Pago'] <= 0.5)
    ]
    grupos = ['Pago seguro','Pago incierto','Alto riesgo']
    data_test['Riesgo'] = np.select(condiciones, grupos, default='No clasificado')
    fig_scatter = px.scatter(
        data_test, x='Score_credito', y='Dias_mora',
        color='Riesgo', size='Monto_pendiente', hover_data=['Prob_Pago'],
        title="Clientes por Score de crédito y Días de mora (tamaño: monto pendiente)"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.markdown("""
    **¿Qué revela este gráfico?**  
    - Los clientes con bajo score y muchos días de mora tienden a ser “alto riesgo”.
    - Puedes segmentar campañas visualmente seleccionando zonas del gráfico.
    """)

# ----------- G3. MAPA DE CALOR DE RIESGO (Monto pendiente vs Score) -----------
with st.expander(" ¿Dónde se concentran los riesgos más altos? (Mapa de calor)"):
    st.info("Visualiza la concentración de riesgo de mora combinando monto pendiente y score de crédito.")
    data_test['Score_bin'] = pd.cut(data_test['Score_credito'], bins=6)
    data_test['Monto_bin'] = pd.cut(data_test['Monto_pendiente'], bins=6)
    # CONVIERTE LOS INTERVALOS A STRING
    data_test['Score_bin_str'] = data_test['Score_bin'].astype(str)
    data_test['Monto_bin_str'] = data_test['Monto_bin'].astype(str)
    pivot_heat = pd.pivot_table(
        data_test, values='Prob_Pago', index='Score_bin_str', columns='Monto_bin_str', aggfunc='mean', observed=False
    )
    fig_heat = px.imshow(
        pivot_heat,
        labels=dict(x="Monto pendiente (bin)", y="Score de crédito (bin)", color="Prob. de Pago"),
        title="Mapa de calor: Riesgo de mora según Score y Monto pendiente"
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    st.markdown("""
    **¿Cómo se usa?**  
    - Zonas azul oscuro = alto riesgo de mora (baja probabilidad de pago).
    - Permite identificar segmentos donde priorizar la gestión de cartera.
    """)

# --- 6. CLASIFICACIÓN DE CLIENTES POR RIESGO (Tabla y Histograma) ---
with st.expander("6️ ¿Cómo se clasifican los clientes según riesgo de pago?"):
    st.info(
        "Se clasifica a los clientes en tres grupos según la probabilidad de pago predicha: "
        "**Pago seguro** (>80%), **Pago incierto** (50-80%), **Alto riesgo** (<50%)."
    )
    fig2 = px.histogram(data_test, x="Prob_Pago", color="Riesgo", barmode="overlay", nbins=30, title="Distribución de riesgo de pago")
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(data_test[['Prob_Pago','Riesgo'] + list(X_test.columns)].sort_values("Prob_Pago",ascending=False).head(10))

    st.markdown("""
    **¿Cómo se usa esto?**
    - Prioriza a los clientes de **alto riesgo** para llamadas personalizadas, acuerdos flexibles o seguimiento intensivo.
    - Clientes de **pago seguro** pueden recibir ofertas/preventas; clientes inciertos requieren monitoreo.
    """)

# --- 7. VISUALIZACIÓN DE COHORTES DE PAGO ---
with st.expander("7️ ¿Cómo evolucionan los pagos por cohortes?"):
    st.info(
        "Aquí puedes analizar el comportamiento de pago según grupos de clientes con características similares (por ejemplo, por días de mora o score de crédito)."
    )

    cohorte = pd.cut(data_test['Dias_mora'], bins=[-1,7,15,30,60], labels=["0-7","8-15","16-30","31-60"])
    res_cohorte = data_test.groupby(cohorte).agg(Pago_real_mean=('Pago_real','mean')).reset_index()
    fig3 = px.bar(res_cohorte, x='Dias_mora', y='Pago_real_mean', labels={'Pago_real_mean':'% Pagos cumplidos'}, title="Pago cumplido por cohorte de días de mora")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("""
    **¿Para qué sirve esto?**
    - Identifica las cohortes con mayor riesgo para acciones preventivas o campañas específicas.
    """)

# --- 8. ANIMACIÓN: TENDENCIA MENSUAL DE COHORTES ---
with st.expander(" ¿Cómo evoluciona el cumplimiento de pago por cohortes y meses? (Animación)"):
    st.info("Visualiza la tendencia mensual del % de pagos cumplidos en distintas cohortes de días de mora.")

    data_full['Cohorte_mora'] = pd.cut(data_full['Dias_mora'], bins=[-1,7,15,30,60], labels=["0-7","8-15","16-30","31-60"])
    resumen_mes = (
        data_full.groupby(['Mes','Cohorte_mora'])
        .agg(pagos_cumplidos=('Pago','mean'))
        .reset_index()
    )

    fig_anim = px.bar(
        resumen_mes, x="Cohorte_mora", y="pagos_cumplidos", color="Cohorte_mora",
        animation_frame="Mes", range_y=[0,1], labels={"pagos_cumplidos":"% Pagos cumplidos"},
        title="Tendencia mensual de cumplimiento de pagos por cohortes de días de mora"
    )
    st.plotly_chart(fig_anim, use_container_width=True)

    st.markdown("""
    **¿Por qué es útil esto?**
    - Permite ver si las estrategias de cobranza están funcionando o si un segmento específico empeora con el tiempo.
    - ¡La animación ayuda a captar la atención de tu audiencia y a explicar la dinámica de la cartera!
    """)

# --- 9. BONUS: INTENCIÓN DE PAGO CON NLP (Simulación Deep Learning) ---
with st.expander("9️ ¿Qué dicen las transcripciones de llamadas? (Simulación Deep Learning NLP)"):
    st.info(
        "Simulamos el uso de modelos NLP para detectar la 'intención de pago' en las transcripciones de llamadas de cobranza."
    )
    intenciones = ["Pagará seguro", "Promete pagar", "Duda en pagar", "No pagará"]
    data_test['Intencion_pago_texto'] = np.random.choice(intenciones, data_test.shape[0])
    fig4 = px.bar(data_test['Intencion_pago_texto'].value_counts(), 
                  x=data_test['Intencion_pago_texto'].value_counts().index, 
                  y=data_test['Intencion_pago_texto'].value_counts().values, 
                  title="Intención de pago detectada en llamadas (simulación NLP)")
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("""
    **¿Por qué es útil esto?**
    - Permite ajustar el mensaje y las acciones del equipo según la intención real expresada por el cliente.
    - Las técnicas de Deep Learning y NLP pueden automatizar este análisis en grandes volúmenes de llamadas.
    """)

st.success("¡Listo! Este demo muestra cómo la IA avanzada anticipa riesgos de mora, ayuda a segmentar y priorizar la gestión de cartera y potencia la cobranza en Telco.")


##streamlit run app_cartera.py
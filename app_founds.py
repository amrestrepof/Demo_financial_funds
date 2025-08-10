# app.py
# Colfondos · Analítica Predictiva + Call Center + Fidelización
# Open rates segmentados por perfil (edad, salario, preferencia, historial)
# Ejecutar: streamlit run app.py

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support, accuracy_score,
    roc_curve, precision_recall_curve, confusion_matrix
)

# --------------------------------
# Configuración de la página
# --------------------------------
st.set_page_config(page_title="Colfondos · Vida Financiera + Call Center", layout="wide")
st.title("Colfondos · Analítica Predictiva de Vida Financiera")
st.caption("Demo con datos simulados — Aporte Voluntario, Planes de Vivienda, Contactabilidad y Fidelización")

# --------------------------------
# 1) Datos simulados (en memoria) con open rates segmentados
# --------------------------------
@st.cache_data(show_spinner=False)
def simulate_data(n=2000, seed=42, segmentado=True):
    rng = np.random.default_rng(seed)

    ids = np.arange(100000, 100000+n)
    edad = rng.integers(24, 62, size=n)

    salario_anterior = np.clip(rng.normal(3.2, 1.1, size=n), 1.2, 15.0) * 1_000_000
    delta_salario = rng.normal(0.08, 0.12, size=n)                # +8% en promedio
    mask_low = rng.random(n) < 0.45                               # muchos con ~0 cambio
    delta_salario[mask_low] = rng.normal(0.0, 0.03, size=mask_low.sum())
    salario_actual = np.clip(salario_anterior * (1 + delta_salario), 1_200_000, 30_000_000)

    cambio_empleador = (rng.random(n) < 0.28).astype(int)
    nit_prev = rng.integers(800_000_000, 900_000_000, size=n)
    nit_act = np.where(cambio_empleador==1, nit_prev + rng.integers(1, 10_000, size=n), nit_prev)

    ahorro_cesantias = rng.gamma(2.0, 3.5, size=n) * 1_000_000
    uso_sim_hip_30d = rng.poisson(0.6, size=n)
    high_intent = rng.random(n) < 0.15
    uso_sim_hip_30d[high_intent] += rng.poisson(3.0, size=high_intent.sum())

    # Preferencia histórica de canal y telefonía
    canal_preferido = rng.choice(["WhatsApp", "Email", "App", "Llamada"], size=n, p=[0.45, 0.25, 0.20, 0.10])
    tiene_telefono = rng.random(n) > 0.06
    telefono = np.where(tiene_telefono, "3" + (rng.integers(10**9, 10**10-1, size=n)).astype(str), "")
    dnc = (rng.random(n) < 0.03).astype(int)  # Do-Not-Call
    consentimiento_datos = (rng.random(n) > 0.02).astype(int)
    hist_llamadas_90d = rng.poisson(1.2, size=n)
    contactos_efectivos_90d = np.minimum(hist_llamadas_90d, rng.poisson(0.7, size=n))
    horario_pref = rng.choice(["Mañana", "Tarde", "Noche"], size=n, p=[0.42, 0.43, 0.15])

    # Afinidad producto (para sesgar engagement): 0 bajo, 1 medio, 2 alto
    afinidad_aporte = (rng.random(n) < 0.25).astype(int) + (rng.random(n) < 0.20).astype(int)
    afinidad_vivienda = (rng.random(n) < 0.20).astype(int) + (rng.random(n) < 0.20).astype(int)

    # ---------------- Open rates segmentados ----------------
    # Bases por canal (promedios Colombia aprox.)
    base_whatsapp = 0.65
    base_email    = 0.22
    base_app      = 0.45

    # Ajustes por perfil
    # Edad: jóvenes abren más app/whatsapp; mayores abren más whatsapp y menos app/email
    adj_age_wa = np.where(edad < 30, +0.08, np.where(edad > 50, +0.03, +0.05))
    adj_age_em = np.where(edad < 30, -0.02, np.where(edad > 50, -0.03, 0.00))
    adj_age_app= np.where(edad < 30, +0.10, np.where(edad > 50, -0.05, +0.02))

    # Salario: mayores ingresos suelen tener hábitos digitales más consistentes (app/email)
    adj_sal_em  = np.where(salario_actual > 10_000_000, +0.05, 0.00)
    adj_sal_app = np.where(salario_actual > 10_000_000, +0.05, 0.00)
    adj_sal_wa  = np.where(salario_actual > 10_000_000, +0.02, 0.00)

    # Preferencia declarada de canal
    adj_pref_wa  = (canal_preferido == "WhatsApp").astype(float) * 0.12
    adj_pref_em  = (canal_preferido == "Email").astype(float) * 0.12
    adj_pref_app = (canal_preferido == "App").astype(float) * 0.12

    # Historial de llamadas: si ha tenido contacto efectivo, sube WA; si no, baja un poco email
    adj_hist_wa = np.clip(contactos_efectivos_90d, 0, 3) * 0.02
    adj_hist_em = np.where(contactos_efectivos_90d == 0, -0.02, 0.00)

    # Afinidad por producto: si afinidad alta con aporte/vivienda, sube WA y App (consumen contenidos)
    adj_aff_wa  = (afinidad_aporte + afinidad_vivienda) * 0.015
    adj_aff_app = (afinidad_aporte + afinidad_vivienda) * 0.02

    # Ruido controlado
    noise_wa  = rng.normal(0, 0.03, size=n)
    noise_em  = rng.normal(0, 0.02, size=n)
    noise_app = rng.normal(0, 0.03, size=n)

    if segmentado:
        open_rate_whatsapp = base_whatsapp + adj_age_wa + adj_sal_wa + adj_pref_wa + adj_hist_wa + adj_aff_wa + noise_wa
        open_rate_email    = base_email    + adj_age_em + adj_sal_em + adj_pref_em + adj_hist_em + noise_em
        open_rate_app      = base_app      + adj_age_app+ adj_sal_app+ adj_pref_app+ adj_aff_app + noise_app
    else:
        # fallback aleatorio simple (no segmentado)
        open_rate_whatsapp = rng.uniform(0.35, 0.95, size=n)
        open_rate_email    = rng.uniform(0.05, 0.60, size=n)
        open_rate_app      = rng.uniform(0.25, 0.85, size=n)

    # Limitar a [0.02, 0.98]
    open_rate_whatsapp = np.clip(open_rate_whatsapp, 0.02, 0.98)
    open_rate_email    = np.clip(open_rate_email,    0.02, 0.98)
    open_rate_app      = np.clip(open_rate_app,      0.02, 0.98)

    # Historial aportes (para modelo A)
    aportes_vol_prev = rng.binomial(1, 0.18, size=n)
    freq_aportes_vol_prev_12m = aportes_vol_prev * rng.poisson(4, size=n)

    # ---------- Etiquetas sintéticas (para entrenar modelos toy) ----------
    # Aporte voluntario: cambio de empleador, delta salario, edad, WA alto, historial aportes
    signal1 = (
        1.5 * cambio_empleador
        + 2.0 * (delta_salario > 0.05).astype(int)
        + 0.8 * ((edad >= 27) & (edad <= 45)).astype(int)
        + 0.5 * (open_rate_whatsapp > 0.6).astype(int)
        + 0.6 * (aportes_vol_prev==1).astype(int)
    )
    prob1 = 1 / (1 + np.exp(-(signal1 - 1.5)))
    y1 = (rng.random(n) < prob1).astype(int)

    # Intención vivienda: uso simulador + cesantías + edad
    signal2 = (
        2.0 * (uso_sim_hip_30d >= 2).astype(int)
        + 1.2 * (ahorro_cesantias > 6_000_000).astype(int)
        + 0.4 * ((edad >= 28) & (edad <= 50)).astype(int)
    )
    prob2 = 1 / (1 + np.exp(-(signal2 - 1.2)))
    y2 = (rng.random(n) < prob2).astype(int)

    # Contactabilidad: teléfono + preferencia llamada + WA + historial − DNC
    sig_c = (
        1.2 * tiene_telefono.astype(int)
        + 1.0 * (canal_preferido == "Llamada").astype(int)
        + 0.9 * (open_rate_whatsapp)       # usar el valor continuo ayuda
        + 0.7 * (contactos_efectivos_90d > 0).astype(int)
        - 2.5 * dnc
    )
    probc = 1 / (1 + np.exp(-(sig_c - 0.9)))
    y_contacta = (rng.random(n) < probc).astype(int)

    df = pd.DataFrame({
        "afiliado_id": ids,
        "edad": edad,
        "salario_anterior": salario_anterior.round(0),
        "salario_actual": salario_actual.round(0),
        "delta_salario": delta_salario,
        "cambio_empleador": cambio_empleador,
        "nit_prev": nit_prev,
        "nit_act": nit_act,
        "ahorro_cesantias": ahorro_cesantias.round(0),
        "uso_simulador_hipoteca_30d": uso_sim_hip_30d,
        "canal_preferido": canal_preferido,
        "open_rate_app": open_rate_app,
        "open_rate_email": open_rate_email,
        "open_rate_whatsapp": open_rate_whatsapp,
        "aportes_voluntarios_prev": aportes_vol_prev,
        "freq_aportes_vol_prev_12m": freq_aportes_vol_prev_12m,
        "afinidad_aporte": afinidad_aporte,
        "afinidad_vivienda": afinidad_vivienda,
        "label_aporte_voluntario": y1,
        "label_plan_vivienda": y2,
        # Call center
        "tiene_telefono": tiene_telefono,
        "telefono": telefono,
        "dnc": dnc,
        "consentimiento_datos": consentimiento_datos,
        "hist_llamadas_90d": hist_llamadas_90d,
        "contactos_efectivos_90d": contactos_efectivos_90d,
        "horario_pref": horario_pref,
        "label_contactabilidad": y_contacta,
    })
    return df

# Parámetros de simulación
segmentado = st.sidebar.toggle("Open rates segmentados por perfil", value=True, help="Si lo apagas, usa open rates aleatorios simples.")
df_full = simulate_data(segmentado=segmentado)
sample = st.sidebar.slider("Tamaño de muestra", 200, len(df_full), 1000, step=100)
df = df_full.sample(sample, random_state=1).reset_index(drop=True)

# --------------------------------
# 2) Explicación ejecutiva
# --------------------------------
with st.expander("ℹ️ ¿Qué predice cada modelo y para qué sirve?", expanded=True):
    st.markdown("""
**Modelo A – Aporte Voluntario**  
Variables: `cambio_empleador`, `delta_salario`, `edad`, `open_rate_whatsapp`, `aportes_voluntarios_prev`.  
**Uso**: disparar WhatsApp/App con proyección de ahorro y CTA a aporte voluntario.

**Modelo B – Planes de Vivienda**  
Variables: `uso_simulador_hipoteca_30d`, `ahorro_cesantias`, `edad`.  
**Uso**: recomendar portafolio conservador, tips tributarios y cobrowsing con asesor.

**Modelo C – Contactabilidad (Call Center)**  
Variables: `tiene_telefono`, `pref_llamada`, `open_rate_whatsapp` (continuo), `contactos_efectivos_90d`, `dnc`.  
**Uso**: priorizar la cola de marcación (a quién llamar primero y en qué franja).
""")

with st.expander("ℹ️ Cómo se simulan los open rates (segmentado)", expanded=True):
    st.markdown("""
Los open rates por canal se ajustan según **edad**, **salario**, **canal preferido**, **historial de contactos** y **afinidad con producto**.  
Ejemplos:
- Jóvenes: ↑ App y WhatsApp; Mayores: ↑ WhatsApp, ↓ App/Email.  
- Preferencia declarada por canal: +12% sobre el canal preferido.  
- Contactos efectivos previos: ↑ WhatsApp.  
- Afinidad (aporte/vivienda) alta: ↑ WhatsApp y App (consumen contenidos).  
""")

# --------------------------------
# 3) Entrenamiento + evaluación
# --------------------------------
def train_and_eval(df, features, label, test_size=0.25, seed=7):
    X = df[features]
    y = df[label]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)
    pipe = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=400))])
    pipe.fit(X_tr, y_tr)
    # Probas
    p_tr = pipe.predict_proba(X_tr)[:, 1]
    p_te = pipe.predict_proba(X_te)[:, 1]
    # Métricas
    def msplit(y_true, p):
        auc = roc_auc_score(y_true, p)
        ap = average_precision_score(y_true, p)
        pred = (p >= 0.5).astype(int)
        acc = accuracy_score(y_true, pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
        return dict(AUC_ROC=auc, AUC_PR=ap, Accuracy=acc, Precision=prec, Recall=rec, F1=f1)
    metrics = {"Train": msplit(y_tr, p_tr), "Test": msplit(y_te, p_te)}
    # Curvas
    fpr, tpr, _ = roc_curve(y_te, p_te); pr_p, pr_r, _ = precision_recall_curve(y_te, p_te)
    # Importancias
    coef_df = pd.DataFrame({"feature": X.columns, "coef_logit": pipe.named_steps["lr"].coef_[0]}).sort_values("coef_logit", ascending=False)
    return pipe, (X_tr, X_te, y_tr, y_te), metrics, coef_df, (fpr, tpr), (pr_r, pr_p)

with st.spinner("Entrenando y evaluando modelos..."):
    featA = ["cambio_empleador", "delta_salario", "edad", "open_rate_whatsapp", "aportes_voluntarios_prev"]
    mA, splitA, metrA, coefA, rocA, prA = train_and_eval(df, featA, "label_aporte_voluntario")

    featB = ["uso_simulador_hipoteca_30d", "ahorro_cesantias", "edad"]
    mB, splitB, metrB, coefB, rocB, prB = train_and_eval(df, featB, "label_plan_vivienda")

    # Contactabilidad: construir features con nombre estable (pref_llamada)
    featC_df = pd.concat([
        df[["tiene_telefono", "contactos_efectivos_90d", "open_rate_whatsapp", "dnc"]],
        (df["canal_preferido"]=="Llamada").astype(int).rename("pref_llamada")
    ], axis=1)
    tmpC = pd.concat([featC_df, df["label_contactabilidad"]], axis=1)
    mC, splitC, metrC, coefC, rocC, prC = train_and_eval(tmpC, ["tiene_telefono","contactos_efectivos_90d","open_rate_whatsapp","dnc","pref_llamada"], "label_contactabilidad")

(XA_tr, XA_te, yA_tr, yA_te) = splitA
(XB_tr, XB_te, yB_tr, yB_te) = splitB
(XC_tr, XC_te, yC_tr, yC_te) = splitC

# --------------------------------
# 4) Scores, triggers, canal y utilidades globales
# --------------------------------
def choose_channel(row):
    rates = {"WhatsApp": row["open_rate_whatsapp"], "Email": row["open_rate_email"], "App": row["open_rate_app"], "Llamada": 0.35}
    return max(rates, key=rates.get)

df["score_aporte_vol"] = mA.predict_proba(df[featA])[:, 1]
df["score_plan_vivienda"] = mB.predict_proba(df[featB])[:, 1]
df["score_contactabilidad"] = mC.predict_proba(featC_df)[:, 1]
df["canal_recomendado"] = df.apply(choose_channel, axis=1)

st.sidebar.subheader("Umbrales")
thrA = st.sidebar.slider("Umbral Aporte Voluntario", 0.10, 0.90, 0.60, 0.05)
thrB = st.sidebar.slider("Umbral Plan Vivienda",   0.10, 0.90, 0.55, 0.05)
thrC = st.sidebar.slider("Umbral Contactabilidad (Call Center)", 0.10, 0.90, 0.55, 0.05)

df["trg_aporte_vol"] = (df["score_aporte_vol"] >= thrA).astype(int)
df["trg_plan_vivienda"] = (df["score_plan_vivienda"] >= thrB).astype(int)
df["trg_contactable"]   = (df["score_contactabilidad"] >= thrC).astype(int)

# KPI top
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Afiliados", len(df))
c2.metric("Triggers aporte voluntario", int(df["trg_aporte_vol"].sum()))
c3.metric("Triggers plan vivienda", int(df["trg_plan_vivienda"].sum()))
c4.metric("Elegibles call (contactables)", int(df["trg_contactable"].sum()))
c5.metric("AUC ROC · Aporte (Test)", f"{metrA['Test']['AUC_ROC']:.2f}")
c6.metric("AUC ROC · Contactabilidad (Test)", f"{metrC['Test']['AUC_ROC']:.2f}")

st.markdown("---")

# --- Elementos compartidos ---
df["motivo_trigger"] = np.select(
    [df["trg_aporte_vol"].eq(1), df["trg_plan_vivienda"].eq(1)],
    ["Aporte voluntario", "Plan vivienda"], default="N/A"
)
df["score_nba"] = df[["score_aporte_vol", "score_plan_vivienda"]].max(axis=1)

def best_hour(hpref):
    return {"Mañana": "09:00-11:30", "Tarde": "14:00-17:00", "Noche": "18:00-20:00"}.get(hpref, "09:00-17:00")
df["mejor_hora_llamar"] = df["horario_pref"].apply(best_hour)

df["prioridad"] = (
    df["score_nba"] * df["score_contactabilidad"] *
    (df["tiene_telefono"].astype(int)) *
    (1 - df["dnc"]) *
    (df["consentimiento_datos"])
)

# --------------------------------
# 5) Tabs (incluye Call Center y Fidelización)
# --------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    [" Triggers", " Vista 360", " What-If", " Métricas y Curvas", " Explicabilidad", " Call Center", " Fidelización & Experiencia"]
)

with tab1:
    st.subheader("Explorador de Triggers")
    tipo = st.selectbox("Tipo de trigger", ["Aporte voluntario", "Plan vivienda"])
    if tipo == "Aporte voluntario":
        tmp = df[df["trg_aporte_vol"]==1].copy()
        tmp["recomendacion"] = "Sugerir aporte voluntario + proyección de ahorro"
    else:
        tmp = df[df["trg_plan_vivienda"]==1].copy()
        tmp["recomendacion"] = "Portafolio conservador + tips tributarios + cobrowsing"
    cols = ["afiliado_id","edad","salario_actual","cambio_empleador","delta_salario",
            "uso_simulador_hipoteca_30d","ahorro_cesantias",
            "score_aporte_vol","score_plan_vivienda","canal_recomendado","recomendacion"]
    st.dataframe(tmp[cols].sort_values(by="afiliado_id").reset_index(drop=True), use_container_width=True)

with tab2:
    st.subheader("Vista 360 del Afiliado")
    sel = st.selectbox("Selecciona un afiliado", df["afiliado_id"])
    row = df[df["afiliado_id"]==sel].iloc[0]

    colA, colB, colC = st.columns(3)
    colA.metric("Edad", int(row["edad"]))
    colA.metric("Salario actual (COP)", f"{int(row['salario_actual']):,}".replace(",", "."))
    colA.metric("Cambio de empleador", "Sí" if row["cambio_empleador"]==1 else "No")

    colB.metric("Score aporte voluntario", f"{row['score_aporte_vol']:.2f}")
    colB.metric("Score plan vivienda", f"{row['score_plan_vivienda']:.2f}")
    colB.metric("Score contactabilidad", f"{row['score_contactabilidad']:.2f}")

    colC.write("**Acciones sugeridas**")
    acc = []
    if row["score_aporte_vol"] >= thrA:
        acc.append("WhatsApp con proyección de ahorro y CTA a aporte voluntario.")
    if row["score_plan_vivienda"] >= thrB:
        acc.append("Simulación hipotecaria + portafolio conservador (beneficios tributarios).")
    if row["score_contactabilidad"] >= thrC and row["tiene_telefono"] and not row["dnc"]:
        acc.append(f"Llamada en franja {row['horario_pref']} (alta prob. contacto).")
    if not acc:
        acc = ["Nurturing por App/Email con contenido educativo."]
    for a in acc: colC.write("• " + a)

with tab3:
    st.subheader("What-If – simulador de efectos")
    sel2 = st.selectbox("Afiliado", df["afiliado_id"], key="whatif")
    r = df[df["afiliado_id"]==sel2].iloc[0]

    st.write("Ajusta variables para simular el impacto en los scores:")
    cambio_emp = st.checkbox("Cambio de empleador", value=bool(r["cambio_empleador"]))
    delta = st.slider("Cambio salarial (%)", -10, 60, int(r["delta_salario"]*100))
    uso_sim = st.slider("Uso de simulador hipotecario (30d)", 0, 10, int(r["uso_simulador_hipoteca_30d"]))
    ces = st.slider("Ahorro cesantías (millones COP)", 0, 50, int(r["ahorro_cesantias"]/1_000_000))

    X1 = pd.DataFrame([{
        "cambio_empleador": int(cambio_emp),
        "delta_salario": delta/100.0,
        "edad": r["edad"],
        "open_rate_whatsapp": r["open_rate_whatsapp"],
        "aportes_voluntarios_prev": r["aportes_voluntarios_prev"]
    }])
    X2 = pd.DataFrame([{
        "uso_simulador_hipoteca_30d": uso_sim,
        "ahorro_cesantias": ces*1_000_000,
        "edad": r["edad"]
    }])

    s1 = float(mA.predict_proba(X1)[0,1]); s2 = float(mB.predict_proba(X2)[0,1])

    col1, col2 = st.columns(2)
    col1.metric("Nuevo score aporte voluntario", f"{s1:.2f}", delta=f"{s1 - r['score_aporte_vol']:+.2f}")
    col2.metric("Nuevo score plan vivienda", f"{s2:.2f}", delta=f"{s2 - r['score_plan_vivienda']:+.2f}")

with tab4:
    st.subheader("Métricas de evaluación y curvas (Test)")
    def metrics_table(m, title):
        st.markdown(f"**{title}**")
        st.dataframe(pd.DataFrame(m).T.style.format("{:.3f}"), use_container_width=True)

    colA, colB, colC = st.columns(3)
    with colA: metrics_table(metrA, "Modelo A · Aporte Voluntario")
    with colB: metrics_table(metrB, "Modelo B · Plan Vivienda")
    with colC: metrics_table(metrC, "Modelo C · Contactabilidad")

    for (name, roc, prc) in [("Aporte", rocA, prA), ("Vivienda", rocB, prB), ("Contactabilidad", rocC, prC)]:
        fig = plt.figure(); plt.plot(roc[0], roc[1]); plt.plot([0,1],[0,1],"--")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC · {name}"); st.pyplot(fig)
        fig = plt.figure(); plt.plot(prc[0], prc[1]); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR · {name}"); st.pyplot(fig)

    yA_pred = (mA.predict_proba(XA_te)[:,1] >= thrA).astype(int)
    yB_pred = (mB.predict_proba(XB_te)[:,1] >= thrB).astype(int)
    yC_pred = (mC.predict_proba(XC_te)[:,1] >= thrC).astype(int)
    cmA = confusion_matrix(yA_te, yA_pred); cmB = confusion_matrix(yB_te, yB_pred); cmC = confusion_matrix(yC_te, yC_pred)

    col1, col2, col3 = st.columns(3)
    with col1: st.write("**Conf. Aporte**"); st.write(pd.DataFrame(cmA, index=["Real 0","Real 1"], columns=["Pred 0","Pred 1"]))
    with col2: st.write("**Conf. Vivienda**"); st.write(pd.DataFrame(cmB, index=["Real 0","Real 1"], columns=["Pred 0","Pred 1"]))
    with col3: st.write("**Conf. Contactabilidad**"); st.write(pd.DataFrame(cmC, index=["Real 0","Real 1"], columns=["Pred 0","Pred 1"]))

with tab5:
    st.subheader("Explicabilidad (coeficientes de la Regresión Logística)")
    col1, col2, col3 = st.columns(3)
    with col1: st.write("**Aporte Voluntario**"); st.dataframe(coefA, use_container_width=True)
    with col2: st.write("**Plan Vivienda**"); st.dataframe(coefB, use_container_width=True)
    with col3: st.write("**Contactabilidad**"); st.dataframe(coefC, use_container_width=True)

with tab6:
    st.subheader("Call Center · Cola de Marcación y Recomendaciones")

    elegibles = df.query("prioridad > 0 and trg_contactable == 1 and tiene_telefono == True and dnc == 0 and consentimiento_datos == 1").copy()

    def script(row):
        if row["trg_aporte_vol"]==1:
            return "Saluda + reconoce cambio laboral; ofrece simulación de aporte voluntario con proyección y beneficios tributarios."
        if row["trg_plan_vivienda"]==1:
            return "Explora intención de vivienda; ofrece asesoría con cobrowsing y portafolio conservador para cesantías."
        return "Diagnóstico breve de necesidades y educación financiera."
    elegibles["guion_sugerido"] = elegibles.apply(script, axis=1)

    calllist = elegibles.sort_values("prioridad", ascending=False)[
        ["afiliado_id","telefono","mejor_hora_llamar","motivo_trigger",
         "score_nba","score_contactabilidad","prioridad","guion_sugerido"]
    ].reset_index(drop=True)

    cA, cB, cC = st.columns(3)
    cA.metric("Registros en cola", len(calllist))
    cB.metric("Contactabilidad promedio", f"{calllist['score_contactabilidad'].mean():.2f}" if len(calllist) else "0.00")
    cC.metric("Conexiones esperadas (∑score_contactabilidad)", f"{calllist['score_contactabilidad'].sum():.1f}" if len(calllist) else "0.0")

    st.dataframe(calllist, use_container_width=True)

    if len(calllist):
        st.download_button(
            label="⬇️ Descargar cola de marcación (CSV)",
            data=calllist.to_csv(index=False).encode("utf-8"),
            file_name="cola_callcenter_colfondos.csv",
            mime="text/csv"
        )

with tab7:
    st.subheader(" Fidelización & Experiencia")

    # 7.1 Fidelización con atención preferencial (alto valor)
    st.markdown("### 1) Fidelización con atención preferencial (clientes de alto valor)")
    df["alto_valor"] = ((df["salario_actual"] > 15_000_000) | (df["score_nba"] > 0.80)).astype(int)
    preferentes = df[df["alto_valor"]==1].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Clientes alto valor", len(preferentes))
    c2.metric("Prom. score NBA (alto valor)", f"{preferentes['score_nba'].mean():.2f}" if len(preferentes) else "0.00")
    c3.metric("Prom. contactabilidad (alto valor)", f"{preferentes['score_contactabilidad'].mean():.2f}" if len(preferentes) else "0.00")

    st.write("**Política sugerida**: SLA prioritario, asignación a agentes senior, canal preferencial y seguimiento ejecutivo.")
    st.dataframe(preferentes[["afiliado_id","salario_actual","score_nba","score_contactabilidad","motivo_trigger","mejor_hora_llamar"]].sort_values("score_nba", ascending=False).head(200), use_container_width=True)

    st.markdown("---")

    # 7.2 Foco Experiencia al Cliente (NPS/CSAT simulados)
    st.markdown("### 2) Foco Experiencia al Cliente")
    rng = np.random.default_rng(123)
    df["NPS"] = rng.integers(-100, 100, size=len(df))
    df["CSAT"] = rng.integers(60, 100, size=len(df))

    colx, coly = st.columns(2)
    with colx:
        st.metric("NPS promedio", f"{df['NPS'].mean():.1f}")
        st.metric("CSAT promedio", f"{df['CSAT'].mean():.1f}")
    with coly:
        st.write("**Distribución NPS / CSAT (promedios)**")
        st.bar_chart(df[["NPS","CSAT"]].mean())

    st.write("**Quick wins experiencia**")
    st.markdown("""
- Mensajes claros sobre beneficios y pasos (reduce confusión y reclamos).
- Citas con cobrowsing para vivienda y decisiones complejas.
- Contenido educativo en App y WhatsApp segmentado por momento de vida.
- Cierre de loop post-gestión (encuesta corta + acción correctiva).
""")

    st.markdown("---")

    # 7.3 Estrategias de contactabilidad outbound
    st.markdown("### 3) Estrategias de contactabilidad para modelo outbound")
    plan_outbound = df[df["trg_contactable"]==1].groupby(["motivo_trigger","canal_recomendado","mejor_hora_llamar"]).size().reset_index(name="volumen")
    st.write("**Segmentación de campañas (trigger, canal, franja):**")
    st.dataframe(plan_outbound.sort_values("volumen", ascending=False), use_container_width=True)

    st.write("**Lineamientos tácticos:**")
    st.markdown("""
- **Priorizar por**: `prioridad = score_nba × score_contactabilidad`, respetando DNC y consentimiento.
- **Secuencia de contacto**: 1º canal preferido → 2º WhatsApp → 3º Llamada en franja óptima.
- **A/B testing**: guiones, CTA, timing y frecuencia; medir *uplift* y conversión por cohorte.
- **Asignación por habilidades**: “vivienda” vs “ahorro” vs “educación financiera”.
""")

# --------------------------------
# 6) Marco ético / regulatorio
# --------------------------------
with st.expander(" Marco ético, regulatorio y tecnológico"):
    st.markdown("""
- **Ética**: transparencia y propósito legítimo; explique al afiliado por qué se le contacta (uso de datos y beneficios).
- **Regulatorio**: respetar **DNC** y **consentimiento**; trazabilidad de predicciones y activaciones por canal.
- **Tecnología**: *feature store*, orquestación de pipelines, activación omnicanal (WhatsApp, App, Email, Call Center) y medición de **uplift**.
- **Operación**: colas por prioridad, ventanas de marcación por franja, asignación por habilidades, control de SLA y *A/B* de guiones.
""")

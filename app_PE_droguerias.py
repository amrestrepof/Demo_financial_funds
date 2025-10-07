import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler 

# --- 1. GENERACIÓN DE DATOS SIMULADOS (30 Agentes en COP) ---
def generar_datos_simulados():
    N_AGENTES = 30
    np.random.seed(42)

    data = {
        'Agente': [f'Agente {i+1:02d}' for i in range(N_AGENTES)],
        'Ventas_Totales': np.random.randint(18_000_000, 34_000_000, N_AGENTES),
        'Gestiones_Totales': np.random.randint(120, 220, N_AGENTES),
        'NPS_Bruto': np.random.randint(30, 70, N_AGENTES) * 10 / 100,
        'TMO_Segundos': np.random.randint(150, 270, N_AGENTES), 
        'Absentismo_Dias': np.random.randint(0, 7, N_AGENTES),
        'Ventas_Alto_Margen': np.random.randint(6_000_000, 20_000_000, N_AGENTES)
    }
    df = pd.DataFrame(data)

    df['TMO_Minutos'] = df['TMO_Segundos'] / 60
    df['TMO_Minutos'] = df['TMO_Minutos'].round(2)
    df['NPS_Agente'] = ((df['NPS_Bruto'] * 10) - 35).round()
    df['Rotacion_Anual'] = np.random.choice([0, 0, 0, 1], size=len(df), p=[0.75, 0.1, 0.1, 0.05])
    
    return df

# --- 2. CÁLCULO DE MÉTRICAS AVANZADAS ---
def calcular_metricas(df):
    df['Ticket_Promedio_Gestion'] = df['Ventas_Totales'] / df['Gestiones_Totales']
    
    max_abs = df['Absentismo_Dias'].max()
    df['Estabilidad'] = (1 - (df['Absentismo_Dias'] / max_abs)) * (1 - df['Rotacion_Anual'])
    
    tmo_range = df['TMO_Minutos'].max() - df['TMO_Minutos'].min()
    nps_range = df['NPS_Agente'].max() - df['NPS_Agente'].min()
    
    tmo_norm = (df['TMO_Minutos'].max() - df['TMO_Minutos']) / tmo_range if tmo_range != 0 else 0
    nps_norm = (df['NPS_Agente'] - df['NPS_Agente'].min()) / nps_range if nps_range != 0 else 0
    
    df['Score_TMO_Optimo'] = (tmo_norm * 0.4) + (nps_norm * 0.6)
    
    return df

# --- FUNCIÓN DE CIENCIA DE DATOS: K-MEANS ---
def clasificar_arquetipos(df_metrics):
    features_cluster = ['Ticket_Promedio_Gestion', 'Score_TMO_Optimo', 'Estabilidad']
    X_cluster = df_metrics[features_cluster].copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_metrics['Cluster'] = kmeans.fit_predict(X_scaled)
    
    cluster_means = df_metrics.groupby('Cluster')[features_cluster].mean().sort_values(by='Ticket_Promedio_Gestion', ascending=True)
    
    mapeo = {
        cluster_means.index[0]: '1. Arquetipo de Riesgo (Desarrollo)',
        cluster_means.index[1]: '2. Arquetipo Central (Sostenimiento)',
        cluster_means.index[2]: '3. Arquetipo de Alto Valor (Mentoría)'
    }
    
    df_metrics['Arquetipo'] = df_metrics['Cluster'].map(mapeo)
    
    return df_metrics

# --- 3. DASHBOARD STREAMLIT (Ajustando formato a COP) ---
def main():
    st.set_page_config(layout="wide", page_title="Demo Analítica de Agentes")
    st.title("Demo: Perfil de Éxito Agentes")
    
    st.header("Sistema de Inteligencia de Operaciones en Ventas para Droguerías")
    st.markdown("---")

    df = generar_datos_simulados()
    df_metrics = calcular_metricas(df)
    df_metrics = clasificar_arquetipos(df_metrics)
    
    # --- 1. PRODUCTIVIDAD Y RENTABILIDAD ---
    st.subheader("1.  Productividad y Rentabilidad (Ventas en COP)")
    
    col1, col2 = st.columns(2)

    with col1:
        max_ticket_agent = df_metrics.loc[df_metrics['Ticket_Promedio_Gestion'].idxmax()]
        st.metric(
            label=" Agente con Mayor Ticket Promedio", 
            value=f"COP {max_ticket_agent['Ticket_Promedio_Gestion']:,.0f}",
            delta=max_ticket_agent['Agente']
        )
        fig_ticket = px.bar(
            df_metrics.sort_values('Ticket_Promedio_Gestion', ascending=False).head(10),
            x='Agente', y='Ticket_Promedio_Gestion',
            title='Top 10 Agentes por Ticket Promedio por Gestión',
            labels={'Ticket_Promedio_Gestion': 'Ticket Promedio (COP)'},
            color='Ticket_Promedio_Gestion', color_continuous_scale=px.colors.sequential.Viridis
        )
        st.plotly_chart(fig_ticket, use_container_width=True)
        
    with col2:
        max_margen_agent = df_metrics.loc[df_metrics['Ventas_Alto_Margen'].idxmax()]
        st.metric(
            label="Agente con Mayor Venta de Alto Margen", 
            value=f"COP {max_margen_agent['Ventas_Alto_Margen']:,.0f}",
            delta=max_margen_agent['Agente']
        )
        fig_margen = px.bar(
            df_metrics.sort_values('Ventas_Alto_Margen', ascending=False).head(10),
            x='Agente', y='Ventas_Alto_Margen',
            title='Top 10 Agentes por Ventas de Alto Margen',
            labels={'Ventas_Alto_Margen': 'Ventas Alto Margen (COP)'},
            color='Ventas_Alto_Margen', color_continuous_scale=px.colors.sequential.Plasma
        )
        st.plotly_chart(fig_margen, use_container_width=True)

    st.markdown("---")
    
    # --- 2. EFICIENCIA Y CALIDAD (TMO ÓPTIMO) ---
    st.subheader("2. Eficiencia Óptima vs. Calidad (TMO en Minutos & NPS)")
    
    fig_tmo_nps = px.scatter(
        df_metrics, x='TMO_Minutos', y='NPS_Agente', color='Score_TMO_Optimo', size='Score_TMO_Optimo', hover_name='Agente', 
        title='TMO (Minutos) vs NPS: ¿Quiénes son los más Eficientes y de Calidad?',
        labels={'TMO_Minutos': 'TMO Promedio (Minutos)', 'NPS_Agente': 'NPS del Agente'},
        color_continuous_scale=px.colors.sequential.Plasma
    )
    fig_tmo_nps.update_layout(xaxis_title="TMO Promedio (Minutos) - Menor es Mejor", yaxis_title="NPS del Agente - Mayor es Mejor")
    st.plotly_chart(fig_tmo_nps, use_container_width=True)
    st.caption("Los Agentes de éxito (puntos grandes y amarillos/blancos) logran **Bajo TMO en Minutos** y Alto NPS. Idealmente, en la esquina inferior derecha.")

    st.markdown("---")

    # --- 3. GESTIÓN DE TALENTO (ESTABILIDAD OPERATIVA) ---
    st.subheader("3. Gestión de Talento (Estabilidad Operativa)")
    st.markdown("La analítica de estabilidad mide el impacto del **Absentismo** y la **Rotación** en los costos de talento y el riesgo operativo.")
    
    col_estabilidad_chart, col_estabilidad_table = st.columns(2)
    
    with col_estabilidad_chart:
        fig_estabilidad = px.bar(
            df_metrics.sort_values('Estabilidad', ascending=False).head(10),
            x='Estabilidad', y='Agente', orientation='h', 
            title='Top 10 Agentes por Índice de Estabilidad',
            labels={'Estabilidad': 'Índice de Estabilidad (Mayor es Mejor)'},
            color='Estabilidad', color_continuous_scale=px.colors.sequential.Greens
        )
        st.plotly_chart(fig_estabilidad, use_container_width=True)
    
    with col_estabilidad_table:
        st.dataframe(
            df_metrics[['Agente', 'Absentismo_Dias', 'Rotacion_Anual', 'Estabilidad']].sort_values('Estabilidad', ascending=False).head(10), 
            hide_index=True, 
            use_container_width=True
        )
        st.caption("Detalle de Estabilidad: La Rotación es 1 si es de alto riesgo de abandono (0 en caso contrario).")

    st.markdown("---")
    
    # --- 4. CIENCIA DE DATOS: ARQUETIPOS DE AGENTES (CLUSTERING) ---
    st.subheader("4. Ciencia de Datos: Arquetipos de Agentes (K-Means)")
    
    col_chart_cluster, col_means_cluster = st.columns(2)
    
    with col_chart_cluster:
        fig_cluster = px.scatter(
            df_metrics, x='Score_TMO_Optimo', y='Ticket_Promedio_Gestion',
            color='Arquetipo', size='Estabilidad', hover_name='Agente',
            title='Arquetipos de Agentes: Rendimiento vs. Eficiencia',
            color_discrete_map={
                '3. Arquetipo de Alto Valor (Mentoría)': 'green',
                '2. Arquetipo Central (Sostenimiento)': 'blue',
                '1. Arquetipo de Riesgo (Desarrollo)': 'red'
            }
        )
        st.plotly_chart(fig_cluster, use_container_width=True)
        st.caption("El tamaño del punto representa la Estabilidad del agente.")
        
    with col_means_cluster:
        # CORRECCIÓN DE FORMATO: Usamos la sintaxis estándar sin el espacio problemático.
        cluster_summary = df_metrics.groupby('Arquetipo')[['Ticket_Promedio_Gestion', 'Score_TMO_Optimo', 'Estabilidad']].mean().sort_values('Ticket_Promedio_Gestion', ascending=False)
        
        st.dataframe(
            cluster_summary.style.format({
                'Ticket_Promedio_Gestion': 'COP {:,.0f}', # Corregido: removido el espacio
                'Score_TMO_Optimo': '{:.2f}',
                'Estabilidad': '{:.2f}'
            }),
            use_container_width=True
        )
        st.caption("""
        **Interpretación Analítica:**
        - **Alto Valor (Verde):** Tienen el mejor rendimiento. Son **Mentores**.
        - **Riesgo (Rojo):** Baja productividad. Necesitan **Desarrollo Intensivo**.
        """)
        
    st.markdown("---")
    
    # --- CONCLUSIÓN FINAL ---
    st.header(" Conclusión Analítica: El Perfil de Éxito No Es Simple")
    
    mejor_agente_ticket_final = df_metrics.loc[df_metrics['Ticket_Promedio_Gestion'].idxmax()]
    
    st.info(
        f"""
        **Demostración del Valor Analítico (Reunión de Capas):**
        1. **Clustering:** El sistema identificó 3 arquetipos, permitiendo dirigir los recursos (mentores/entrenamiento) de forma precisa.
        2. **Rentabilidad:** El **Agente {mejor_agente_ticket_final['Agente']}** es el más rentable, con un Ticket Promedio de **COP {mejor_agente_ticket_final['Ticket_Promedio_Gestion']:,.0f}** por gestión.
        
        Esto demuestra que el éxito es un **vector de tres dimensiones (Rentabilidad, Eficiencia, Estabilidad)** que solo la analítica avanzada puede medir.
        """
    )
    st.markdown("---")
    st.caption("¡Contáctenos para transformar sus datos en decisiones estratégicas!")

if __name__ == "__main__":
    main()
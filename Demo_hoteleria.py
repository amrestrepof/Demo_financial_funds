import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import collections

# --- Configuración Inicial ---
# Descargar el léxico de VADER para el análisis de sentimientos
# Esto solo se ejecuta una vez si no está presente
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    # Usamos st.spinner para que se vea mejor mientras descarga
    with st.spinner("Descargando recursos para análisis de sentimiento (VADER)..."):
        nltk.download('vader_lexicon')

# Inicializar el analizador de sentimientos
sia = SentimentIntensityAnalyzer()

# --- Datos de Muestra (Simulación) ---
# ¡AHORA EN ESPAÑOL!
MOCK_TRANSCRIPT = [
    {'speaker': 'Agente', 'text': 'Buenos días, ¿hablo con el Sr. Pérez?'},
    {'speaker': 'Cliente', 'text': 'Sí, con él.'},
    {'speaker': 'Agente', 'text': '¡Excelente! Mi nombre es Alex, le llamo de Hoteles Colsubsidio. Tenemos una oferta exclusiva para nuestros afiliados.'},
    {'speaker': 'Cliente', 'text': '¿Ah sí? ¿Qué tipo de oferta?'},
    {'speaker': 'Agente', 'text': 'Es una tarifa especial de temporada baja para nuestros hoteles Peñalisa y Colonial. ¿Es usted afiliado activo y mayor de edad?'},
    {'speaker': 'Cliente', 'text': 'Sí, lo soy. Pero usualmente estoy muy ocupado para viajar.'},
    {'speaker': 'Agente', 'text': 'Entiendo. Esta oferta es para viajar de domingo a jueves, perfecto para evitar multitudes. El plan de Peñalisa incluye desayuno buffet.'},
    {'speaker': 'Cliente', 'text': 'Eso suena... interesante. Pero Peñalisa es demasiado caliente para mí.'},
    {'speaker': 'Agente', 'text': '¡No hay problema! También tenemos el Hotel Colonial en Paipa, con plan de pensión completa y acceso a piscinas termales.'},
    {'speaker': 'Cliente', 'text': '¡Wow, eso sí suena fantástico! Me encanta Paipa.'},
    {'speaker': 'Agente', 'text': '¡Maravilloso! Puedo enviarle la cotización detallada ahora mismo. ¿Cuál es su correo?'},
    {'speaker': 'Cliente', 'text': 'Es ejemplo@mail.com. Esto fue de gran ayuda, gracias.'}
]

# --- Definición de Tópicos (Voz del Cliente) ---
# ¡AHORA EN ESPAÑOL!
TOPIC_KEYWORDS = {
    'Hotel Peñalisa': ['peñalisa', 'caliente', 'girardot', 'calor'],
    'Hotel Colonial': ['colonial', 'paipa', 'termales', 'piscinas'],
    'Precio/Oferta': ['oferta', 'tarifa', 'precio', 'cotización', 'ayuda', 'gracias'],
    'Disponibilidad': ['ocupado', 'viajar', 'tiempo', 'domingo', 'jueves', 'multitudes']
}

# --- Funciones Auxiliares ---

def get_sentiment_label(text):
    """
    Analiza el texto y devuelve una etiqueta de sentimiento (Positivo, Negativo, Neutral)
    y el puntaje 'compound'.
    MODIFICADO para incluir reglas simples en español, ya que VADER es para inglés.
    """
    text_lower = text.lower()

    # --- REGLAS EN ESPAÑOL ---
    if any(p in text_lower for p in ['fantástico', 'excelente', 'me encanta', 'me gusta', 'maravilloso', 'gran ayuda', 'perfecto', 'sí suena']):
        return 'Positivo', 0.9
    if any(n in text_lower for n in ['demasiado caliente', 'no me gusta', 'muy ocupado', 'problema', 'pero']):
        # Damos un puntaje negativo leve, 'pero' es una objeción
        return 'Negativo', -0.4
    # --- Fin Reglas ---

    # Fallback a VADER (probablemente será Neutral)
    score = sia.polarity_scores(text)['compound']
    if score > 0.1:
        return 'Positivo', score
    elif score < -0.1:
        return 'Negativo', score
    else:
        return 'Neutral', score

def get_sentiment_color(label):
    """Devuelve un color para la etiqueta de sentimiento."""
    if label == 'Positivo':
        return 'green'
    elif label == 'Negativo':
        return 'red'
    else:
        return 'gray'

# --- NUEVA FUNCIÓN: Detección de Intención ---
def get_intent(text):
    """
    Analiza el texto y devuelve una etiqueta de intención simple.
    """
    text_lower = text.lower()
    
    if '?' in text or any(w in text_lower for w in ['qué', 'cómo', 'cuál', 'cuándo']):
        return 'Pregunta'
    if any(w in text_lower for w in ['pero', 'demasiado caliente', 'muy ocupado', 'no me gusta']):
        return 'Objeción'
    if any(w in text_lower for w in ['fantástico', 'me encanta', 'me gusta', 'maravilloso', 'perfecto', 'sí suena']):
        return 'Interés Alto'
    if any(w in text_lower for w in ['sí, con él', 'sí, lo soy', 'ejemplo@mail.com', 'gracias']):
        return 'Confirmación/Cierre'
    return 'Declaración'

# --- Función: Análisis de Tópicos ---
def find_topics(text):
    """
    Analiza el texto del cliente y extrae tópicos basados en palabras clave.
    """
    found_topics = []
    text_lower = text.lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                if topic not in found_topics: # Evitar tópicos duplicados en la misma línea
                    found_topics.append(topic)
    return found_topics

# --- Interfaz de Streamlit ---
st.set_page_config(layout="wide", page_title="Speech Analytics Demo")

st.title("Demo de Speech Analytics 📊")
st.markdown("Análisis de sentimiento en llamadas (Agente vs. Cliente)")

# --- MODIFICADO: Simulación de audio con un botón ---
st.info("Presione el botón para simular el análisis de una llamada de muestra.")

if st.button("▶️ Iniciar Simulación de Análisis"):
    # Mensaje de éxito
    st.success("Simulación de análisis de audio completada. Mostrando resultados...")
    st.markdown("---")

    # Listas para almacenar datos para los gráficos
    agent_sentiments = []
    client_sentiments = []
    client_topics_list = []  # Para el gráfico de tópicos
    client_key_moments = [] # Para extraer frases clave
    full_transcript_data = []
    
    # --- NUEVO: Diccionario para Evaluación de Calidad (QA) ---
    agent_scorecard = {
        "Saludo Inicial": False,
        "Perfilamiento Comercial": False,
        "Manejo de Objeción (Retención)": False,
        "Amabilidad del Agente": False
    }
    client_amabilidad_detected = False

    # Columnas para la visualización
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Transcripción y Análisis de Sentimiento")
        
        for line in MOCK_TRANSCRIPT:
            speaker = line['speaker']
            text = line['text']
            sentiment_label, sentiment_score = get_sentiment_label(text)
            intent_label = get_intent(text) # Obtener intención
            text_lower = text.lower() # Para QA
            
            # --- Análisis de Tópicos ---
            topics_found = []
            if speaker == 'Cliente':
                topics_found = find_topics(text)
                client_topics_list.extend(topics_found) # Para el gráfico resumen
                
                # --- NUEVO: Capturar momentos clave ---
                if intent_label == 'Objeción' or intent_label == 'Interés Alto':
                    client_key_moments.append({
                        'intent': intent_label, 
                        'text': text, 
                        'sentiment': sentiment_label
                    })
                
                # --- NUEVO: Check de Amabilidad del Cliente ---
                if not client_amabilidad_detected and any(kw in text_lower for kw in ['gracias', 'gran ayuda', 'fantástico', 'me encanta']):
                    client_amabilidad_detected = True
            
            # --- NUEVO: Lógica de Evaluación de Calidad (Agente) ---
            if speaker == 'Agente':
                if not agent_scorecard["Saludo Inicial"] and any(kw in text_lower for kw in ['buenos días', 'buenas tardes', 'le llamo de']):
                    agent_scorecard["Saludo Inicial"] = True
                if not agent_scorecard["Perfilamiento Comercial"] and any(kw in text_lower for kw in ['afiliado activo', 'mayor de edad']):
                    agent_scorecard["Perfilamiento Comercial"] = True
                if not agent_scorecard["Manejo de Objeción (Retención)"] and any(kw in text_lower for kw in ['no hay problema', 'también tenemos', 'entiendo']):
                    agent_scorecard["Manejo de Objeción (Retención)"] = True
                if not agent_scorecard["Amabilidad del Agente"] and any(kw in text_lower for kw in ['excelente', 'maravilloso', 'perfecto', 'me alegra']):
                    agent_scorecard["Amabilidad del Agente"] = True
            
            # Recopilar datos
            full_transcript_data.append({
                'Orador': speaker,
                'Texto': text,
                'Sentimiento': sentiment_label,
                'Intención': intent_label, # <-- NUEVO
                'Puntaje': sentiment_score,
                'Tópicos': ", ".join(topics_found)
            })
            
            if speaker == 'Agente':
                agent_sentiments.append(sentiment_score)
                speaker_color = "blue"
            else:
                client_sentiments.append(sentiment_score)
                speaker_color = "green"

            # --- Mostrar la línea de transcripción con tópicos e intención ---
            sentiment_color = get_sentiment_color(sentiment_label)
            
            # Crear "badges" para los tópicos
            topic_html = ""
            if topics_found:
                badges = "".join([f'<span style="background-color: #eee; color: #333; padding: 2px 8px; border-radius: 10px; font-size: 0.8em; margin-right: 5px;">{t}</span>' for t in topics_found])
                topic_html = f"<div style='margin-top: 5px;'><strong>Tópicos:</strong> {badges}</div>"

            # --- MODIFICADO: Añadido badge de Intención ---
            st.markdown(
                f"""
                <div style="border-left: 4px solid {speaker_color}; padding-left: 10px; margin-bottom: 15px; border-radius: 5px; background-color: #f9f9f9; padding: 8px;">
                    <strong>{speaker}:</strong> "{text}"<br>
                    <small style="color:{sentiment_color};"><strong>Sentimiento:</strong> {sentiment_label} ({sentiment_score:.2f})</small>
                    | <small><strong>Intención:</strong> {intent_label}</small>
                    {topic_html}
                </div>
                """,
                unsafe_allow_html=True
            )

    with col2:
        st.subheader("Métricas de la Llamada")
        
        # Calcular métricas (simuladas y reales)
        # En un sistema real, los tiempos vendrían del análisis de audio
        total_lines_agent = len(agent_sentiments)
        total_lines_client = len(client_sentiments)
        avg_sentiment_agent = sum(agent_sentiments) / total_lines_agent if total_lines_agent > 0 else 0
        avg_sentiment_client = sum(client_sentiments) / total_lines_client if total_lines_client > 0 else 0

        st.metric(label="Duración de la llamada (simulado)", value="01:05 min")
        st.metric(label="Tiempo de habla del Agente (simulado)", value="60%")
        st.metric(label="Tiempo de habla del Cliente (simulado)", value="40%")
        
        # --- NUEVO: Sección de Evaluación de Calidad (Agente) ---
        st.markdown("---")
        st.subheader("Evaluación de Calidad (Agente)")
        
        def display_scorecard_item(label, success):
            """Función para mostrar un ítem del scorecard con ícono."""
            if success:
                st.markdown(f"✅ **{label}:** Detectado")
            else:
                st.markdown(f"❌ **{label}:** No Detectado")

        display_scorecard_item("Saludo Inicial", agent_scorecard["Saludo Inicial"])
        display_scorecard_item("Perfilamiento Comercial", agent_scorecard["Perfilamiento Comercial"])
        display_scorecard_item("Manejo de Objeción (Retención)", agent_scorecard["Manejo de Objeción (Retención)"])
        display_scorecard_item("Amabilidad del Agente", agent_scorecard["Amabilidad del Agente"])
        
        # --- NUEVO: Amabilidad del Cliente ---
        st.markdown("---")
        st.subheader("Voz del Cliente (General)")
        display_scorecard_item("Amabilidad del Cliente", client_amabilidad_detected)


        st.markdown("---")
        st.subheader("Resumen de Sentimiento")

        # Mostrar puntajes promedio
        st.metric(
            label="Sentimiento Promedio del Agente",
            value=f"{avg_sentiment_agent:.2f}",
            help="Positivo > 0.1, Negativo < -0.1"
        )
        st.metric(
            label="Sentimiento Promedio del Cliente",
            value=f"{avg_sentiment_client:.2f}",
            help="Positivo > 0.1, Negativo < -0.1"
        )

        # Gráfico de Sentimiento
        st.markdown("##### Sentimiento por Orador (Promedio)")
        chart_data = pd.DataFrame({
            'Orador': ['Agente', 'Cliente'],
            'Puntaje Promedio': [avg_sentiment_agent, avg_sentiment_client]
        })
        st.bar_chart(chart_data.set_index('Orador'))

        # Gráfico de distribución de sentimientos
        st.markdown("##### Distribución de Sentimientos (Líneas de diálogo)")
        df_transcript = pd.DataFrame(full_transcript_data)
        sentiment_counts = df_transcript.groupby('Orador')['Sentimiento'].value_counts().unstack(fill_value=0)
        st.bar_chart(sentiment_counts)
        
        # --- Gráfico de Voz del Cliente (Tópicos) ---
        st.markdown("---")
        st.subheader("Voz del Cliente (Tópicos Mencionados)")
        if client_topics_list:
            topic_counts = collections.Counter(client_topics_list)
            df_topics = pd.DataFrame.from_dict(topic_counts, orient='index', columns=['Menciones'])
            df_topics = df_topics.sort_values(by='Menciones', ascending=False)
            st.bar_chart(df_topics)
        else:
            st.info("No se detectaron tópicos clave en el cliente.")
            
        # --- NUEVO: Sección de Momentos Clave ---
        st.markdown("---")
        st.subheader("Momentos Clave (Voz del Cliente)")
        if client_key_moments:
            for moment in client_key_moments:
                if moment['intent'] == 'Objeción':
                    st.error(
                        f"**Objeción Detectada (Sentimiento: {moment['sentiment']}):**\n"
                        f'"{moment["text"]}"'
                    )
                elif moment['intent'] == 'Interés Alto':
                    st.success(
                        f"**Interés Alto Detectado (Sentimiento: {moment['sentiment']}):**\n"
                        f'"{moment["text"]}"'
                    )
        else:
            st.info("No se detectaron objeciones o momentos de interés alto.")

    st.markdown("---")
    st.subheader("Datos Completos de la Transcripción")
    st.dataframe(pd.DataFrame(full_transcript_data))
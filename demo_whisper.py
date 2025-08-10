import streamlit as st
import whisper
import tempfile
import os


# Tu ruta de ffmpeg y modelo Whisper
os.environ["PATH"] += os.pathsep + r"C:\Users\arestrepo\Downloads\ffmpeg-7.0.2-essentials_build\ffmpeg-7.0.2-essentials_build\bin"
# Diccionario con la descripción de cada modelo
modelos = {
    "tiny": "Muy rápido, baja precisión. Ideal para equipos con pocos recursos.",
    "base": "Rápido y más preciso que tiny. Buen balance entre velocidad y calidad.",
    "small": "Más preciso que base, velocidad aceptable. Recomendado para tareas generales.",
    "medium": "Alta precisión, más lento. Útil en aplicaciones exigentes.",
    "large": "Máxima precisión, más lento y pesado. Ideal para resultados de alta calidad."
}

# Sidebar
st.sidebar.title("Configuración")
modelo = st.sidebar.selectbox(
    "Selecciona el modelo de Whisper:",
    options=list(modelos.keys()),
    format_func=lambda x: x.capitalize()
)
st.sidebar.info(modelos[modelo])

st.title("Prototipo de Transcripción de Audios con Whisper")
st.write("Selecciona el modelo y sube tu audio para transcribirlo.")

# Subida de archivo
audio_file = st.file_uploader("Carga un archivo de audio (.mp3, .wav, etc.)", type=["mp3", "wav", "m4a", "flac", "aac"])

if audio_file:
    # Tomar la extensión del archivo
    ext = os.path.splitext(audio_file.name)[-1]
    if ext == "":
        ext = ".mp3"  # Por defecto mp3 si no tiene extensión

    # Guardar archivo temporalmente
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    # Transcribir con Whisper
    st.info("Transcribiendo, esto puede tomar unos segundos...")
    model = whisper.load_model(modelo)
    result = model.transcribe(tmp_path)

    # Mostrar resultados
    st.subheader("Transcripción:")
    st.success(result["text"])

    # Borrar archivo temporal
    os.remove(tmp_path)

st.markdown("---")
st.markdown("""
**Modelos disponibles:**
- **Tiny:** Muy rápido, baja precisión. Ideal para equipos con pocos recursos.
- **Base:** Rápido y más preciso que tiny. Buen balance entre velocidad y calidad.
- **Small:** Más preciso que base, velocidad aceptable. Recomendado para tareas generales.
- **Medium:** Alta precisión, más lento. Útil en aplicaciones exigentes.
- **Large:** Máxima precisión, más lento y pesado. Ideal para resultados de alta calidad.
""")
## streamlit run demo_whisper.py
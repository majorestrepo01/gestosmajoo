import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
import platform

# Configuración inicial de la página
st.set_page_config(page_title="Reconocimiento de Gestos", page_icon="👍", layout="centered")

# Cargar modelo con caché para optimizar el rendimiento
@st.cache_resource
def cargar_modelo():
    return load_model('keras_model.h5')

model = cargar_modelo()

# Barra lateral informativa
with st.sidebar:
    st.title("ℹ️ Información")
    st.write("Modelo entrenado con **Teachable Machine**.")
    st.caption(f"Versión de Python: {platform.python_version()}")

# Encabezado principal
st.title("👍👎 Reconocimiento de Gestos")
st.write("Captura una foto con tu cámara para detectar si tienes el **pulgar arriba** o **pulgar abajo**.")

# Cargar imagen decorativa principal (si existe)
try:
    image = Image.open('OIG5.jpg')
    st.image(image, use_column_width=True)
except FileNotFoundError:
    pass

st.markdown("---")

# Captura de cámara
img_file_buffer = st.camera_input("Toma una foto para analizar")

if img_file_buffer is not None:
    # Preprocesamiento de la imagen
    img = Image.open(img_file_buffer).convert('RGB')
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized)

    # Normalización (-1 a 1) requerida por Teachable Machine
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1.0
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Predicción
    with st.spinner('Analizando imagen...'):
        prediction = model.predict(data)

    prob_arriba = float(prediction[0][0])
    prob_abajo = float(prediction[0][1])

    st.markdown("### Resultado del Análisis")

    # Mapeo de predicciones basadas en labels.txt
    if prob_arriba > prob_abajo:
        st.success(f"👍 **Pulgar arriba** (Confianza: {prob_arriba * 100:.1f}%)")
        st.progress(prob_arriba)
    else:
        st.error(f"👎 **Pulgar abajo** (Confianza: {prob_abajo * 100:.1f}%)")
        st.progress(prob_abajo)

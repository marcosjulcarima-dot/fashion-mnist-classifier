
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

st.set_page_config(
    page_title="Fashion MNIST Classifier",
    layout="wide"
)

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

@st.cache_resource
def load_trained_model():
    return tf.keras.models.load_model("modelo_final.h5", compile=False)

@st.cache_data
def load_dataset():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    return X_train, y_train, X_test, y_test

model = load_trained_model()
X_train, y_train, X_test, y_test = load_dataset()

st.title("Clasificador de Prendas - Fashion MNIST")
st.write(
    "Aplicación interactiva desarrollada con Streamlit para clasificar imágenes de prendas "
    "utilizando un modelo de Deep Learning entrenado con Fashion MNIST."
)

tab1, tab2, tab3 = st.tabs(["Dataset", "Resultados", "Predicción"])

with tab1:
    st.header("Información del Dataset")

    st.write("""
    Fashion MNIST es un dataset compuesto por imágenes en escala de grises de 28x28 píxeles.
    El objetivo es clasificar cada imagen dentro de una de las 10 categorías de prendas.
    Este problema corresponde a una tarea de clasificación multiclase.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Características principales")
        st.write(f"*Número de imágenes de entrenamiento:* {X_train.shape[0]:,}")
        st.write(f"*Número de imágenes de prueba:* {X_test.shape[0]:,}")
        st.write("*Tamaño de cada imagen:* 28 x 28 píxeles")
        st.write("*Tipo de imagen:* Escala de grises")
        st.write("*Número de clases:* 10")

    with col2:
        st.subheader("Tabla de clases")
        clases_df = pd.DataFrame({
            "ID": list(range(10)),
            "Clase": CLASS_NAMES
        })
        st.dataframe(clases_df, use_container_width=True)

    st.subheader("Ejemplos de imágenes por clase")

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.ravel()

    for i in range(10):
        idx = np.where(y_train == i)[0][0]
        axes[i].imshow(X_train[idx], cmap="gray")
        axes[i].set_title(CLASS_NAMES[i])
        axes[i].axis("off")

    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Distribución de clases")

    conteo = pd.Series(y_train).value_counts().sort_index()
    conteo_df = pd.DataFrame({
        "Clase": CLASS_NAMES,
        "Cantidad": conteo.values
    })

    st.bar_chart(conteo_df.set_index("Clase"))

    st.write("""
    La distribución muestra que las clases se encuentran balanceadas, con aproximadamente
    la misma cantidad de imágenes por categoría. Esto reduce el riesgo de que el modelo
    favorezca una clase específica durante el entrenamiento.
    """)

with tab2:
    st.header("Resultados del Modelo")

    st.subheader("Tabla de métricas de los tres modelos")

    try:
        metricas = pd.read_csv("metricas_modelos.csv")
        st.dataframe(metricas, use_container_width=True)

        st.subheader("Comparación de Accuracy en Test")
        st.bar_chart(metricas.set_index("Modelo")["Accuracy"])

        mejor_modelo = metricas.sort_values("Accuracy", ascending=False).iloc[0]
        st.success(
            f"El mejor modelo fue *{mejor_modelo['Modelo']}*, "
            f"con un accuracy de *{mejor_modelo['Accuracy']:.4f}*."
        )

    except Exception as e:
        st.warning("No se pudo cargar el archivo metricas_modelos.csv.")
        st.write(e)

    st.subheader("Matriz de Confusión del Mejor Modelo")

    try:
        st.image(
            "matriz_confusion.png",
            caption="Matriz de confusión - CNN + Regularización",
            use_container_width=True
        )
    except Exception as e:
        st.warning("No se pudo cargar matriz_confusion.png.")
        st.write(e)

    st.subheader("Curvas de Entrenamiento del Mejor Modelo")

    try:
        st.image(
            "curvas_entrenamiento.png",
            caption="Curvas de accuracy y loss - CNN + Regularización",
            use_container_width=True
        )
    except Exception as e:
        st.warning("No se pudo cargar curvas_entrenamiento.png.")
        st.write(e)

    st.write("""
    Los resultados muestran que la CNN con regularización obtuvo el mejor desempeño general.
    Este modelo combina capas convolucionales con Batch Normalization y Dropout, lo que permite
    conservar la estructura espacial de las imágenes y reducir el sobreajuste observado en la CNN base.
    """)

with tab3:
    st.header("Clasificador Interactivo")

    st.write("""
    Sube una imagen de una prenda en formato JPG, PNG o BMP. La imagen será convertida
    a escala de grises, redimensionada a 28x28 píxeles y normalizada antes de realizar
    la predicción.
    """)

    uploaded_file = st.file_uploader(
        "Sube una imagen de prenda",
        type=["jpg", "jpeg", "png", "bmp"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L")
        image_resized = image.resize((28, 28))

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Imagen procesada")
            st.image(
                image_resized,
                width=220,
                caption="Imagen en escala de grises redimensionada a 28x28"
            )

        arr = np.array(image_resized).astype("float32") / 255.0
        arr = arr.reshape(1, 28, 28, 1)

        pred = model.predict(arr)
        predicted_class = int(np.argmax(pred))
        confidence = float(np.max(pred))

        probs_df = pd.DataFrame({
            "Clase": CLASS_NAMES,
            "Probabilidad": pred[0]
        }).sort_values("Probabilidad", ascending=False)

        with col2:
            st.subheader("Resultado de la predicción")
            st.write(f"*Predicción:* {CLASS_NAMES[predicted_class]}")
            st.write(f"*Confianza:* {confidence:.2%}")

            st.subheader("Top 3 clases más probables")
            st.dataframe(probs_df.head(3), use_container_width=True)

        st.subheader("Probabilidades para las 10 clases")
        st.bar_chart(probs_df.set_index("Clase"))

    else:
        st.info("Carga una imagen para obtener una predicción.")

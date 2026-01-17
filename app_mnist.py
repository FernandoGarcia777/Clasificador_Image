import gradio as gr
import numpy as np
from PIL import Image, ImageOps
import joblib

# 1. CARGAR EL MODELO ENTRENADO
# Asegúrate de tener el archivo .pkl en la misma carpeta o ajusta la ruta
try:
    modelo = joblib.load('modelo_mnist.pkl')
    print("✅ Modelo cargado correctamente.")
except:
    print("⚠️ No se encontró 'modelo_mnist.pkl'. El código funcionará pero simulará predicciones.")
    modelo = None

def preprocesar_imagen(imagen_usuario):
    """
    Toma la imagen dibujada por el usuario y la transforma al formato
    que espera el modelo entrenado con MNIST (28x28, fondo negro, aplanada).
    """
    if imagen_usuario is None:
        return None

    # CORRECCIÓN PARA GRADIO 4.0+:
    # Si la imagen viene como un diccionario (con capas), tomamos solo la imagen final ('composite')
    if isinstance(imagen_usuario, dict):
        imagen_usuario = imagen_usuario['composite']

    # A. Convertir a imagen de PIL y a escala de grises ('L')
    # La imagen viene de Gradio como un array numpy
    img = Image.fromarray(imagen_usuario).convert('L')
    
    # B. Invertir colores (Dibujo: negro sobre blanco -> MNIST: blanco sobre negro)
    img = ImageOps.invert(img)
    
    # C. Redimensionar a 28x28 píxeles (usando resampleo de alta calidad)
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    
    # D. Convertir a array de Numpy
    img_array = np.array(img)
    
    # E. Aplanar la imagen (de 28x28 a 1x784)
    img_flatten = img_array.reshape(1, -1)
    
    return img_flatten

def predecir_numero(imagen):
    # 1. Preprocesar
    datos_procesados = preprocesar_imagen(imagen)
    
    if datos_procesados is None:
        return "Dibuja algo primero."
    
    # 2. Predecir usando el modelo real
    if modelo:
        prediccion = modelo.predict(datos_procesados)[0]
        return f"El modelo predice que es un: {prediccion}"
    else:
        return "Modo prueba (Modelo no cargado): Imagen procesada correctamente."

# 3. INTERFAZ GRÁFICA CON GRADIO
# 'sketchpad' permite dibujar. 'label' muestra el texto.
demo = gr.Interface(
    fn=predecir_numero, 
    inputs="sketchpad", 
    outputs="label",
    title="Clasificador de Dígitos MNIST",
    description="Dibuja un número del 0 al 9 en el recuadro y el modelo adivinará cuál es."
)

# Ejecutar la app
if __name__ == "__main__":
    demo.launch(share=True)
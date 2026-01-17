[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FernandoGarcia777//Clasificador_Image/blob/main/Demo.ipynb)
# 🖊️ MNIST Handwritten Digit Classifier

## 📋 Descripción
Este proyecto implementa un sistema de Visión por Computadora para reconocer dígitos escritos a mano (0-9) utilizando el famoso dataset **MNIST**. Fue desarrollado siguiendo el Capítulo 3 de *"Hands-On Machine Learning"* de Aurélien Géron.

El objetivo fue construir un clasificador robusto sin utilizar Redes Neuronales Convolucionales (CNNs), exprimiendo al máximo los algoritmos clásicos y técnicas de preprocesamiento.

## 🛠 Tecnologías y Librerías
* **Python 3.x**
* **Scikit-Learn:** `KNeighborsClassifier`, `GridSearchCV`.
* **Procesamiento de Imágenes:** `scipy.ndimage` (para Data Augmentation/Shift).
* **Visualización:** Matplotlib (para visualizar la Matriz de Confusión).

## ⚙️ Enfoque Técnico
1.  **Exploración:** Visualización de los dígitos (imágenes de 28x28 píxeles aplanadas).
2.  **Data Augmentation:** Implementé una función personalizada para desplazar las imágenes (arriba, abajo, izquierda, derecha) y expandir el set de entrenamiento, lo que mejoró la generalización del modelo.
3.  **Selección de Modelo:** Se utilizó `KNeighborsClassifier` (KNN) por su efectividad en este tipo de patrones.
4.  **Optimización:** Ajuste de hiperparámetros (`n_neighbors`, `weights`) logrando una precisión superior al **97%**.

## 📊 Resultados y Métricas
* **Accuracy (Test Set):** 97.XX%
* **Matriz de Confusión:** Análisis de errores comunes (ej. el modelo confunde el 5 con el 3).
* *(Opcional: Puedes poner aquí una imagen de tu matriz de confusión)*

## 📂 Estructura del repositorio
* `mnist_classifier.ipynb`: Notebook principal con todo el flujo de trabajo.
* `utils.py`: Funciones auxiliares para graficar y aumentar datos.

---
*Proyecto realizado con fines educativos para dominar los fundamentos de clasificación de imágenes y validación cruzada.*

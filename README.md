# TextLab — Machine Learning Experiment Studio

Plataforma web para configurar, ejecutar y evaluar experimentos de clasificación supervisada de textos. Desarrollada en Python para el MiniHackathon de Aprendizaje Automático.

**Aplicación publicada:** https://textlab-minihackathon-qcbv8iraguvzajqnnsq2qy.streamlit.app/ 

**Repositorio:** https://github.com/IvanIchin/textlab-minihackathon

**Tablero de seguimiento:** https://github.com/users/IvanIchin/projects/1/views/1 

## Funcionalidades

- Carga de archivos CSV etiquetados, selección de columnas, análisis de clases, detección de nulos y duplicados y limpieza opcional.
- Extracción de n-gramas de palabras, etiquetas POS, stopwords y content words; tamaños de 1, 2 o 3 y representaciones de frecuencia, binaria o TF-IDF.
- Entrenamiento de regresión logística, SVM, Random Forest y Naive Bayes multinomial; proporción train/test, hiperparámetros y validación cruzada estratificada opcional de 3, 5 o 10 folds.
- Evaluación mediante accuracy, precision macro, recall macro, F1 macro, matriz de confusión y gráficas por clase y validación cruzada.
- Historial de experimentos durante la sesión y predicción de textos individuales o CSV sin etiquetar, con descarga de predicciones.


## Uso de TextLab

1. Cargar un CSV etiquetado en **Configuración y Resultados** y elegir las columnas de texto y etiqueta.
2. Revisar la calidad y distribución de clases; seleccionar las opciones de limpieza.
3. Configurar las características, tamaño del n-grama y representación.
4. Elegir el clasificador, sus parámetros, proporción de entrenamiento y validación cruzada.
5. Seleccionar **Entrenar modelo** y revisar las métricas y gráficas.
6. Abrir **Comparación y Predicción**, consultar el historial y elegir un modelo entrenado para predecir un texto o archivo CSV.

## Pipeline

La división train/test es estratificada. El extractor de características y el clasificador forman un `Pipeline` de Scikit-learn. La validación cruzada, cuando se activa, opera exclusivamente sobre el conjunto de entrenamiento; el vectorizador se ajusta en cada fold de entrenamiento. El pipeline final se ajusta sobre train y evalúa sobre test.

## Estructura

```text
textlab-minihackathon/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
└── src/
    ├── __init__.py
    ├── data_manager.py
    ├── feature_extractor.py
    ├── model_trainer.py
    ├── evaluator.py
    └── experiment_hub.py
```

## Consideraciones

- Los experimentos se conservan durante la sesión, no en una base de datos persistente.
- Se requieren datos suficientes por clase para la división estratificada y los folds elegidos.
- Las stopwords se configuran en español; el etiquetador POS predeterminado de NLTK está orientado al inglés.
- La tabla de historial conserva las características principales y métricas, pero no registra explícitamente todos los hiperparámetros y ajustes de validación.

## Integrantes

- Iván Alexander Pérez Ichin 
- María Fernanda Estrada Gutiérrez 
- José Gabriel Hernández Pérez 
- Ricardo Caleb Morales Juarez

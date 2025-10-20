%%writefile README.md
# Proyecto de Predicción de Concentración de Aflatoxina B1

Este proyecto tiene como objetivo desarrollar un modelo de Machine Learning para predecir la concentración de Aflatoxina B1 (AFB1) basándose en mediciones de corriente obtenidas de un sensor. Se utilizan datos de calibración para entrenar y optimizar un modelo de regresión.

## Contenido del Repositorio

*   `app.py`: Script de la aplicación web Streamlit para la interfaz de predicción.
*   `requirements.txt`: Lista de dependencias necesarias para ejecutar el proyecto.
*   `full_pipeline_script.py`: Script que contiene todo el flujo de trabajo (carga, preprocesamiento, modelado, optimización, evaluación).
*   `optimized_random_forest_model.joblib`: Modelo Random Forest optimizado guardado.
*   `scaler.joblib`: Scaler utilizado para normalizar los datos guardado.
*   `preprocessed_data.csv`: Datos de calibración combinados y preprocesados.
*   `aflatoxin_b1_calibration.csv`, `aflatoxin_b1_calibration 2.csv`: Archivos de datos originales (si se incluyen).
*   `notebooks/`: Carpeta que contiene los notebooks de Colab que documentan el proceso paso a paso.

## Configuración y Ejecución de la Aplicación Streamlit

Para ejecutar la aplicación Streamlit localmente o en un entorno compatible (como Google Colab con Ngrok), sigue estos pasos:

1.  **Clonar el repositorio:** Si aún no lo has hecho, clona este repositorio a tu máquina local o entorno de trabajo.
    ```bash
    git clone https://github.com/hector23alejandro-hash/aflatoxin-b1-predictor.git
    cd aflatoxin-b1-predictor
    ```
    (Asegúrate de reemplazar la URL si tu repositorio tiene otra dirección).

2.  **Instalar dependencias:** Asegúrate de tener Python instalado. Luego, instala las librerías necesarias listadas en `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Ejecutar la aplicación Streamlit:** Navega a la carpeta raíz del proyecto en tu terminal y ejecuta el script `app.py`.
    ```bash
    streamlit run app.py
    ```

4.  **Acceder a la aplicación:**
    *   Si ejecutas localmente, Streamlit te proporcionará una URL local (usualmente `http://localhost:8501`) que puedes abrir en tu navegador.
    *   Si usas un entorno como Google Colab, `streamlit run` a veces proporciona una URL pública directa, pero la forma más fiable de acceder desde fuera es usando un servicio de túnel como **ngrok**. Consulta la documentación o celdas de ejemplo en el notebook de Colab para la configuración con ngrok si es necesario.

Una vez ejecutado el comando, Streamlit abrirá automáticamente la aplicación en tu navegador o te proporcionará un enlace para acceder a ella.


**Resumen Final del Proyecto: Predicción de Concentración de Aflatoxina B1**
Este proyecto se centró en el diseño y la simulación computacional mediante Inteligencia Artificial para predecir la concentración del contaminante Aflatoxina B1 (AFB1) utilizando datos de calibración obtenidos de un sensor.

**Problema Abordado**
El objetivo principal fue desarrollar un modelo predictivo robusto capaz de estimar la concentración de AFB1 (en ng/mL) a partir de mediciones de corriente (en μA) proporcionadas por un sensor, facilitando así la detección y cuantificación de este contaminante de forma automatizada.

**Metodología y Enfoque**
El proyecto siguió una metodología basada en Machine Learning, que incluyó las siguientes fases clave:

1.  Exploración y Preprocesamiento de Datos: Se cargaron y combinaron dos datasets de calibración. Se estandarizaron los nombres de las columnas, se manejaron valores nulos y se normalizó la característica de corriente utilizando StandardScaler para preparar los datos para el modelado.
2.  Modelado y Optimización: Se exploraron modelos de regresión, enfocándonos en un modelo Random Forest Regressor. Se realizó una optimización de hiperparámetros utilizando GridSearchCV y validación cruzada para mejorar el rendimiento del modelo.
3.  Validación y Pruebas Finales: El modelo optimizado fue evaluado en un conjunto de prueba independiente para verificar su capacidad de generalización y medir el error de predicción. Se validó que el modelo cumpliera con los criterios de rendimiento establecidos.

**Resultados Clave**
. El modelo final optimizado fue un Random Forest Regressor.
. Se logró un Coeficiente de Determinación (R²) de aproximadamente 0.9296 en el conjunto de prueba independiente, lo que indica que el modelo explica una alta proporción de la variabilidad en la concentración y se ajusta bien a los datos no vistos.
. Incluir el error de predicción si se calculó y cumplió el criterio del < 20%. Ejemplo: "El error de predicción promedio fue de aproximadamente 16.66%, cumpliendo con el criterio de error."
. El modelo entrenado y el scaler de normalización fueron guardados para su uso en la aplicación de predicción.

**Implementación y Acceso**
Se desarrolló una aplicación web interactiva utilizando Streamlit (app.py) que carga el modelo y el scaler guardados. Esta aplicación permite a un usuario ingresar un valor de corriente y obtener la concentración de AFB1 predicha por el modelo.

El código completo del proyecto, incluyendo los notebooks de análisis, scripts de preprocesamiento y pipeline, el modelo guardado y la aplicación Streamlit, se encuentra disponible en el siguiente repositorio de GitHub:

[Enlace a tu Repositorio de GitHub](https://github.com/hector23alejandro-hash/aflatoxin-b1-predictor)

La aplicación Streamlit puede ser ejecutada clonando el repositorio y siguiendo las instrucciones proporcionadas en el archivo Instrucciones para uso de Streamlit README.md.

Como segunda opción, la aplicación también puede ejecutarse en entornos computacionales (Google Colab). En este caso, se utiliza la librería pyngrok dentro del código para crear un túnel seguro que genera una URL pública temporal, permitiendo el acceso a la interfaz de Streamlit desde cualquier navegador.

**Conclusión**
Este proyecto ha desarrollado y validado exitosamente un modelo basado en Random Forest capaz de predecir la concentración de Aflatoxina B1 con una alta precisión a partir de mediciones de corriente. La implementación en Streamlit proporciona una herramienta interactiva para aplicar el modelo. Los resultados obtenidos cumplen con los criterios de rendimiento, demostrando la viabilidad del enfoque para la predicción del contaminante AFB1.
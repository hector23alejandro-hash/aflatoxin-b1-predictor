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
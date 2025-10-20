
## Resultados Semana 5 - Meta 5: Validación del Modelo Random Forest

**Objetivo:** Validar el rendimiento del modelo Random Forest optimizado en un conjunto de datos no visto (el conjunto de prueba) y verificar que el error de predicción promedio sea menor al 20%.

**Pasos Realizados:**

1.  **Carga del Modelo Optimizado:** Se cargó exitosamente el modelo Random Forest previamente guardado (`optimized_random_forest_model.joblib`).
2.  **Carga y División de Datos:** Se cargaron los datos preprocesados (`preprocessed_data.csv`) y se dividieron en conjuntos de entrenamiento y prueba, utilizando la misma división que en las etapas anteriores para asegurar consistencia.
3.  **Predicciones en el Conjunto de Prueba:** Se utilizaron las características del conjunto de prueba (`X_test`) como entrada para el modelo cargado, obteniendo las predicciones de concentración (`y_pred_optimized_rf`).
4.  **Cálculo de Errores de Predicción:** Se calcularon métricas de error para cuantificar la diferencia entre las predicciones y los valores reales del conjunto de prueba (`y_test`).
    *   Error Absoluto Medio (MAE): 1.6935
    *   Error Cuadrático Medio (MSE): 8.1608
    *   Raíz del Error Cuadrático Medio (RMSE): 2.8567
5.  **Cálculo del Error Porcentual Promedio:** Se calculó el error de predicción promedio como un porcentaje de la media de los valores reales del conjunto de prueba.
    *   Error de Predicción Promedio (basado en MAE): 16.66%
6.  **Verificación del Criterio de Error (< 20%):** Se comparó el error de predicción porcentual con el umbral del 20%.

**Hallazgos Clave:**

*   El modelo Random Forest optimizado fue cargado y utilizado exitosamente para hacer predicciones.
*   Las métricas de error (MAE, MSE, RMSE) proporcionan una medida cuantitativa del rendimiento del modelo en el conjunto de prueba.
*   El error de predicción promedio calculado fue de aproximadamente **16.66%**.

**Conclusión:**

El criterio para la Meta 5, que requería un error de predicción promedio menor al 20%, fue **cumplido exitosamente**, ya que el error obtenido (**16.66%**) es inferior a 20%. Esto indica que el modelo optimizado tiene una buena capacidad para predecir la concentración de Aflatoxina B1 en datos no vistos dentro de un margen de error aceptable para esta etapa.

**Próximos Pasos:**

Continuar con las siguientes metas del proyecto, que incluyen:

*   **Meta 6:** Desarrollar la interfaz web con Streamlit y subir todos los archivos a GitHub.
*   **Meta 7:** Documentar el proyecto, realizar pruebas finales y preparar la entrega.

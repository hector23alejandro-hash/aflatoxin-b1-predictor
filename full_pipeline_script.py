import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os

print("--- Script Completo del Procedimiento de Calibración y Modelado ---")
print("Este script realiza la carga, preprocesamiento, modelado, optimización y evaluación.")

# Rutas a los archivos de datos originales (asume que están en /content/)
file_path_1 = "/content/aflatoxin_b1_calibration.csv"
file_path_2 = "/content/aflatoxin_b1_calibration 2.csv"
preprocessed_data_path = "/content/preprocessed_data.csv" # Ruta para guardar/cargar datos preprocesados
optimized_model_path = "/content/optimized_random_forest_model.joblib" # Ruta para guardar modelo
scaler_path = "/content/scaler.joblib" # Ruta para guardar scaler


# =============================================================================
# FASE 1: Exploración y Preprocesamiento de Datos (Combinado de Metas 1 y 2)
# =============================================================================

print("\n=== FASE 1: Exploración y Preprocesamiento ===")

# 1. Cargar Datasets
print("1. 📂 Cargando datasets originales...")
try:
    df1 = pd.read_csv(file_path_1)
    print(f"   - {os.path.basename(file_path_1)} cargado. Dimensiones: {df1.shape}")

    df2 = pd.read_csv(file_path_2)
    print(f"   - {os.path.basename(file_path_2)} cargado. Dimensiones: {df2.shape}")
except FileNotFoundError as e:
    print(f"❌ Error cargando archivos: {e}")
    print("   Asegúrate de que los archivos CSV originales estén en '/content/'. Saliendo del script.")
    exit() # Exit if essential files are not found
except Exception as e:
     print(f"❌ Ocurrió un error inesperado al cargar archivos: {e}. Saliendo del script.")
     exit()


# 2. Estandarizar Columnas y Combinar
print("\n2. 🔄 Estandarizando columnas y combinando datasets...")
# Automatically detect potential concentration and current columns
column_mapping_df1 = {}
for col in df1.columns:
    lower_col = col.lower()
    if 'concentracion' in lower_col or 'concentration' in lower_col or 'x' == lower_col.strip():
        column_mapping_df1[col] = 'concentration'
    elif 'corriente' in lower_col or 'current' in lower_col or 'y' == lower_col.strip():
        column_mapping_df1[col] = 'current'

column_mapping_df2 = {}
for col in df2.columns:
    lower_col = col.lower()
    if 'concentracion' in lower_col or 'concentration' in lower_col or 'x' == lower_col.strip():
        column_mapping_df2[col] = 'concentration'
    elif 'corriente' in lower_col or 'current' in lower_col or 'y' == lower_col.strip():
        column_mapping_df2[col] = 'current'

if len(column_mapping_df1) == 2 and len(column_mapping_df2) == 2:
    df1 = df1.rename(columns=column_mapping_df1)
    df2 = df2.rename(columns=column_mapping_df2)
    df_combined = pd.concat([df1, df2], ignore_index=True)
    print(f"   ✅ Datasets combinados. Dimensiones: {df_combined.shape}")
else:
    print("   ❌ Error: No se pudieron identificar las columnas necesarias en uno o ambos datasets originales. Saliendo del script.")
    exit()


# 3. Limpieza de Datos
print("\n3. 🧹 Limpieza de datos...")
df_clean = df_combined.dropna().copy()
df_clean['concentration'] = pd.to_numeric(df_clean['concentration'], errors='coerce')
df_clean['current'] = pd.to_numeric(df_clean['current'], errors='coerce')
df_clean.dropna(subset=['concentration', 'current'], inplace=True)
print(f"   ✅ Datos limpios y numéricos: {df_clean.shape[0]} filas")


# 4. Normalización de Características
print("\n4. 📏 Normalización de características (Current)...")
if not df_clean.empty:
    X = df_clean[['current']].values # Use original current for fitting scaler
    y = df_clean['concentration'].values

    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)

    print("   ✅ Características 'current' normalizadas.")

    # Guardar el scaler para uso futuro (en la app Streamlit)
    try:
        joblib.dump(scaler, scaler_path)
        print(f"   💾 Scaler guardado como '{os.path.basename(scaler_path)}'")
    except Exception as e:
        print(f"   ❌ Error al guardar el scaler: {e}")

    # Crear DataFrame con datos preprocesados (current original y concentration)
    # Y luego con current normalizado y concentration para el modelado
    df_preprocessed_full = df_clean.copy() # Keep original current
    df_preprocessed_full['current_normalized'] = X_normalized.flatten() # Add normalized current

    # Guardar el dataset preprocesado completo (opcional, pero útil)
    try:
        df_preprocessed_full.to_csv(preprocessed_data_path, index=False)
        print(f"   💾 Datos preprocesados guardados como '{os.path.basename(preprocessed_data_path)}'")
    except Exception as e:
        print(f"   ❌ Error al guardar los datos preprocesados: {e}")

    # Datos para modelado (característica normalizada, variable objetivo)
    X_modeling = df_preprocessed_full[['current_normalized']].values
    y_modeling = df_preprocessed_full['concentration'].values

else:
    print("   ⚠️ No hay datos limpios y numéricos para normalizar. Saliendo del script.")
    exit()


# 5. División de Datos (para Modelado)
print("\n5. ✂️ División de datos para modelado...")
if X_modeling.shape[0] > 1 and y_modeling.shape[0] > 1: # Ensure enough samples for split
    X_train, X_test, y_train, y_test = train_test_split(
        X_modeling, y_modeling, test_size=0.2, random_state=42, shuffle=True
    )
    print(f"   📚 Training: {X_train.shape[0]} muestras")
    print(f"   🧪 Testing: {X_test.shape[0]} muestras")
else:
    print("   ⚠️ No hay suficientes datos para dividir en entrenamiento y prueba. Saliendo del script.")
    exit()


# =============================================================================
# FASE 2: Modelado y Optimización (Combinado de Metas 3 y 4)
# =============================================================================

print("\n=== FASE 2: Modelado y Optimización (Random Forest) ===")

# 6. Configurar y Realizar Optimización de Hiperparámetros (Random Forest)
print("6. ⚙️ Optimizando hiperparámetros de Random Forest con validación cruzada...")

# Definir el espacio de búsqueda para GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Estrategia de validación cruzada
cv_strategy = KFold(n_splits=min(5, X_train.shape[0]), shuffle=True, random_state=42) # Use min(5, n_samples)

# Inicializar el modelo base (no es necesario optimizar LR aquí, solo RF)
rf_base_model = RandomForestRegressor(random_state=42)

# Configurar GridSearchCV
grid_search = GridSearchCV(estimator=rf_base_model, param_grid=param_grid, cv=cv_strategy, scoring='r2', n_jobs=-1)

try:
    grid_search.fit(X_train, y_train)
    print("   ✅ Búsqueda de hiperparámetros completada.")
    print(f"   🏆 Mejores hiperparámetros: {grid_search.best_params_}")
    print(f"   📈 Mejor R² en validación cruzada: {grid_search.best_score_:.4f}")

except Exception as e:
    print(f"   ❌ Ocurrió un error durante la búsqueda de hiperparámetros: {e}. Saliendo del script.")
    exit()

# 7. Entrenar el Modelo Final con los Mejores Hiperparámetros
print("\n7. 🏋️ Entrenando modelo Random Forest final con mejores hiperparámetros...")
final_rf_model = RandomForestRegressor(**grid_search.best_params_, random_state=42)

try:
    final_rf_model.fit(X_train, y_train) # Entrenar en todo el conjunto de entrenamiento
    print("   ✅ Entrenamiento del modelo final completado.")
except Exception as e:
     print(f"   ❌ Ocurrió un error durante el entrenamiento del modelo final: {e}. Saliendo del script.")
     exit()


# =============================================================================
# FASE 3: Validación y Guardado (Combinado de Meta 5 y parte de Meta 6)
# =============================================================================

print("\n=== FASE 3: Validación y Guardado ===")

# 8. Evaluar el Modelo Final en el Conjunto de Prueba
print("8. 📊 Evaluando modelo final en el conjunto de prueba...")
try:
    y_pred_final = final_rf_model.predict(X_test)
    r2_test_final = r2_score(y_test, y_pred_final)
    mae_test_final = mean_absolute_error(y_test, y_pred_final)

    print(f"   ✅ R² en conjunto de prueba: {r2_test_final:.4f}")
    print(f"   ✅ MAE en conjunto de prueba: {mae_test_final:.4f}")

    # Calcular error porcentual promedio
    if np.mean(y_test) != 0:
        percentage_error = (mae_test_final / np.mean(y_test)) * 100
        print(f"   ✅ Error de Predicción Promedio (MAE como % de la media real): {percentage_error:.2f}%")
        # Verificar criterio < 20%
        if percentage_error < 20:
            print("   ✅ Criterio de error < 20% CUMPLIDO.")
        else:
             print("   ⚠️ Criterio de error < 20% NO CUMPLIDO.")
    else:
        print("   ⚠️ No se puede calcular el error porcentual: la media de los valores reales es cero.")


except Exception as e:
    print(f"   ❌ Ocurrió un error durante la evaluación en el conjunto de prueba: {e}. Saliendo del script.")
    exit()

# 9. Guardar el Modelo Optimizado
print("\n9. 💾 Guardando el modelo Random Forest optimizado...")
try:
    joblib.dump(final_rf_model, optimized_model_path)
    print(f"   ✅ Modelo guardado exitosamente como '{os.path.basename(optimized_model_path)}'")
except Exception as e:
    print(f"   ❌ Ocurrió un error al guardar el modelo: {e}. Saliendo del script.")
    exit()


print("\n--- Script Completado ---")
print("✅ Proceso de calibración y modelado finalizado. Modelo y scaler guardados.")

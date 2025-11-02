%%writefile app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Define the paths for the model and scaler files
# *** MODIFIED PATHS TO BE RELATIVE TO THE REPOSITORY ROOT ***
MODEL_PATH = 'optimized_random_forest_model.joblib'

# --- Load Model ---
try:
    optimized_rf_model = joblib.load(MODEL_PATH)
    st.success(f"✅ Modelo '{os.path.basename(MODEL_PATH)}' cargado exitosamente.")
except FileNotFoundError:
    st.error(f"❌ Error: Modelo no encontrado en {MODEL_PATH}. Asegúrate de que el archivo '{os.path.basename(MODEL_PATH)}' esté en la raíz de tu repositorio de GitHub.")
    st.stop() # Stop execution if model not found
except Exception as e:
    st.error(f"❌ Error cargando el modelo: {e}")
    st.stop()

# --- Load or Recreate Scaler ---
# Ideally, the scaler should be saved during preprocessing.
# If it wasn't, we need to load the original data and fit a new scaler.
# This assumes the original files are available in the repository root.
PREPROCESSED_DATA_PATH = 'preprocessed_data.csv' # Assuming preprocessed data might also be in root
SCALER_PATH = 'scaler.joblib' # Assuming scaler might be saved in root

try:
    if os.path.exists(SCALER_PATH): # Check if scaler was saved
         scaler = joblib.load(SCALER_PATH)
         st.success("✅ Scaler cargado exitosamente.")
    # If scaler wasn't saved, load original data to fit it
    # Assuming original files are available in the repository root
    elif os.path.exists('aflatoxin_b1_calibration.csv') and os.path.exists('aflatoxin_b1_calibration 2.csv'):
        st.warning(f"⚠️ Scaler no encontrado en '{SCALER_PATH}'. Refitando scaler desde archivos de datos originales.")
        try:
            df1_orig = pd.read_csv('aflatoxin_b1_calibration.csv')
            df2_orig = pd.read_csv('aflatoxin_b1_calibration 2.csv')

            # Standardize column names (must match preprocessing step)
            # Use robust column finding like in the full pipeline script
            def standardize_columns(df):
                 column_mapping = {}
                 for col in df.columns:
                     lower_col = col.lower()
                     if 'concentracion' in lower_col or 'concentration' in lower_col or 'x' == lower_col.strip():
                         column_mapping[col] = 'concentration'
                     elif 'corriente' in lower_col or 'current' in lower_col or 'y' == lower_col.strip():
                         column_mapping[col] = 'current'
                 if len(column_mapping) == 2:
                     return df.rename(columns=column_mapping)
                 else:
                     return None # Indicate failure to find columns

            df1_orig = standardize_columns(df1_orig)
            df2_orig = standardize_columns(df2_orig)

            if df1_orig is not None and df2_orig is not None:
                df_combined_orig = pd.concat([df1_orig, df2_orig], ignore_index=True)
                df_combined_orig = df_combined_orig.dropna()
                df_combined_orig['current'] = pd.to_numeric(df_combined_orig['current'], errors='coerce')
                df_combined_orig.dropna(subset=['current'], inplace=True)

                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                # Fit on the original 'current' column from the combined data
                if not df_combined_orig[['current']].empty:
                     scaler.fit(df_combined_orig[['current']].values)
                     st.success("✅ Scaler refitado desde datos originales.")

                     # Optional: Save the refitted scaler for future use in deployment environment
                     # This might require appropriate write permissions, which can be tricky in some deployment envs.
                     # joblib.dump(scaler, SCALER_PATH)
                     # st.info(f"Scaler refitado guardado como '{SCALER_PATH}' (si el entorno lo permite).")

                else:
                    st.error("❌ Error: No hay datos de 'current' válidos en los archivos originales para refitar el scaler.")
                    st.stop()


            else:
                st.error("❌ Error: No se pudieron identificar las columnas necesarias en los archivos de datos originales para refitar el scaler.")
                st.stop()


        except FileNotFoundError: # This block is inside the elif, so this FileNotFoundError is redundant here
             pass # Handled by the outer elif
        except Exception as e:
            st.error(f"❌ Ocurrió un error refitando el scaler desde datos originales: {e}")
            st.stop()

    elif os.path.exists(PREPROCESSED_DATA_PATH):
         st.warning(f"⚠️ Scaler no encontrado en '{SCALER_PATH}'. Intentando cargar datos preprocesados desde '{PREPROCESSED_DATA_PATH}' para refitar.")
         try:
              df_preprocessed = pd.read_csv(PREPROCESSED_DATA_PATH)
              if 'current' in df_preprocessed.columns: # Need original current to fit scaler
                  from sklearn.preprocessing import StandardScaler
                  scaler = StandardScaler()
                  if not df_preprocessed[['current']].empty:
                       scaler.fit(df_preprocessed[['current']].values) # Fit on the original 'current' column
                       st.success("✅ Scaler refitado desde datos preprocesados (columna 'current').")
                  elif 'current_normalized' in df_preprocessed.columns and 'concentration' in df_preprocessed.columns:
                       # If only normalized current is available, cannot refit the scaler correctly.
                       st.error(f"❌ Error: El archivo '{PREPROCESSED_DATA_PATH}' solo contiene 'current_normalized', no 'current' original para refitar el scaler.")
                       st.stop()
                  else:
                       st.error(f"❌ Error: El archivo '{PREPROCESSED_DATA_PATH}' no contiene la columna 'current' original para refitar el scaler.")
                       st.stop()

              else:
                   st.error(f"❌ Error: El archivo '{PREPROCESSED_DATA_PATH}' no contiene la columna 'current' original para refitar el scaler.")
                   st.stop()


         except FileNotFoundError: # This block is inside the elif, redundant
              pass # Handled by the outer elif
         except Exception as e:
              st.error(f"❌ Ocurrió un error refitando el scaler desde datos preprocesados: {e}")
              st.stop()

    else:
        st.error(f"❌ Error: Scaler no encontrado en '{SCALER_PATH}' y tampoco se encontraron los archivos de datos originales ('aflatoxin_b1_calibration.csv', 'aflatoxin_b1_calibration 2.csv') ni '{PREPROCESSED_DATA_PATH}' en la raíz de tu repositorio para refitarlo.")
        st.stop()

except Exception as e:
    st.error(f"❌ Ocurrió un error al cargar o refitar el scaler: {e}")
    st.stop()


# --- Streamlit App Interface ---
st.title('🔬 Predictor de Concentración de Aflatoxina B1')
st.write('Ingresa el valor de corriente (μA) medido por el sensor para predecir la concentración de Aflatoxina B1 (ng/mL).')

# Input field for Current
current_input = st.number_input('Valor de Corriente (μA)', min_value=0.0, format="%.6f")

# Prediction Button
if st.button('Predecir Concentración'):
    if optimized_rf_model and scaler: # Ensure model and scaler were loaded/refitted successfully
        try:
            # Prepare the input for prediction: reshape and scale
            # The model was trained on the *normalized* current
            input_data = np.array([[current_input]])
            normalized_current = scaler.transform(input_data)

            # Make prediction
            predicted_concentration = optimized_rf_model.predict(normalized_current)

            # Display the prediction
            st.subheader('Resultado de la Predicción:')
            st.success(f'La concentración de Aflatoxina B1 predicha es: **{predicted_concentration[0]:.4f} ng/mL**')

        except Exception as e:
            st.error(f"❌ Ocurrió un error durante la predicción: {e}")
    else:
        st.warning("⚠️ Modelo o Scaler no cargados correctamente. No se puede realizar la predicción.")

# Optional: Display information about the loaded model/scaler
st.sidebar.header("Información del Modelo")
if optimized_rf_model:
    st.sidebar.write(f"- Modelo cargado: {os.path.basename(MODEL_PATH)}")
    st.sidebar.write(f"- Tipo de Modelo: {type(optimized_rf_model).__name__}")
if scaler:
     st.sidebar.write(f"- Scaler: {type(scaler).__name__}")
     # st.sidebar.write(f"- Scaler Mean: {scaler.mean_[0]:.4f}") # Uncomment if you want to show scaler details
     # st.sidebar.write(f"- Scaler Std Dev: {scaler.scale_[0]:.4f}")

st.sidebar.write("\n")
st.sidebar.info("Esta aplicación utiliza un modelo Random Forest entrenado para predecir la concentración de Aflatoxina B1 a partir de mediciones de corriente.")

# Note: This cell will block execution as long as the streamlit app and ngrok tunnel are running.
# To stop, interrupt the cell execution.

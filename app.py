import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Define the paths for the model and scaler files
MODEL_PATH = '/content/optimized_random_forest_model.joblib'
# Assuming the scaler was saved during preprocessing or can be recreated
# For simplicity here, we'll demonstrate how to load if it was saved.
# If the scaler was NOT saved, you would need to refit it using
# the training data or ensure your prediction function handles scaling internally.
# Since the preprocessed data is available, we can refit the scaler if needed.

# --- Load Model ---
try:
    optimized_rf_model = joblib.load(MODEL_PATH)
    st.success(f"✅ Modelo '{os.path.basename(MODEL_PATH)}' cargado exitosamente.")
except FileNotFoundError:
    st.error(f"❌ Error: Modelo no encontrado en {MODEL_PATH}. Asegúrate de que se guardó correctamente.")
    st.stop() # Stop execution if model not found
except Exception as e:
    st.error(f"❌ Error cargando el modelo: {e}")
    st.stop()

# --- Load or Recreate Scaler ---
# Ideally, the scaler should be saved during preprocessing.
# If it wasn't, we need to load the preprocessed data and fit a new scaler.
# This assumes the preprocessed data file contains the 'current_normalized' column
# used to fit the original scaler.
PREPROCESSED_DATA_PATH = '/content/preprocessed_data.csv'
try:
    if os.path.exists('/content/scaler.joblib'): # Check if scaler was saved
         scaler = joblib.load('/content/scaler.joblib')
         st.success("✅ Scaler cargado exitosamente.")
    elif os.path.exists(PREPROCESSED_DATA_PATH):
        st.warning(f"⚠️ Scaler no encontrado. Refitando scaler desde '{os.path.basename(PREPROCESSED_DATA_PATH)}'.")
        df_preprocessed = pd.read_csv(PREPROCESSED_DATA_PATH)
        if 'current_normalized' in df_preprocessed.columns and 'concentration' in df_preprocessed.columns:
             # Need the ORIGINAL (unnormalized) data to fit the scaler correctly
             # This requires access to the original combined data BEFORE normalization.
             # A robust solution would save the scaler object itself.
             # For this example, we'll use a placeholder and note the requirement.
             # In a real scenario, you MUST save the scaler used during training.
             # Assuming 'current' column was present before normalization step in preprocessed data or load original.
             # Let's assume we can reload the original data to fit the scaler for demonstration.
             # A more robust approach is to save the scaler object alongside the model.

             # --- Alternative: If scaler wasn't saved, load original data to fit it ---
             # This assumes original files are available or a combined original df was saved.
             # Let's try loading original data assuming they are still in /content
             try:
                 df1_orig = pd.read_csv('/content/aflatoxin_b1_calibration.csv')
                 df2_orig = pd.read_csv('/content/aflatoxin_b1_calibration 2.csv')

                 # Standardize column names (must match preprocessing step)
                 df1_orig = df1_orig.rename(columns={'x (Concentracion)': 'concentration', 'y (Corriente)': 'current'})
                 df2_orig = df2_orig.rename(columns={'X (Concentracion)': 'concentration', 'Y (Corriente)': 'current'})

                 df_combined_orig = pd.concat([df1_orig, df2_orig], ignore_index=True)
                 df_combined_orig = df_combined_orig.dropna()
                 df_combined_orig['current'] = pd.to_numeric(df_combined_orig['current'], errors='coerce')
                 df_combined_orig.dropna(subset=['current'], inplace=True)


                 from sklearn.preprocessing import StandardScaler
                 scaler = StandardScaler()
                 scaler.fit(df_combined_orig[['current']].values) # Fit on the original 'current' column
                 st.success("✅ Scaler refitado desde datos originales.")

                 # Optional: Save the refitted scaler for future use
                 joblib.dump(scaler, '/content/scaler.joblib')
                 st.info("Scaler refitado guardado como '/content/scaler.joblib'")


             except FileNotFoundError:
                  st.error("❌ Error: Archivos de datos originales no encontrados para refitar el scaler.")
                  st.stop()
             except Exception as e:
                 st.error(f"❌ Error refitando el scaler: {e}")
                 st.stop()
        else:
             st.error(f"❌ Error: El archivo '{os.path.basename(PREPROCESSED_DATA_PATH)}' no contiene las columnas necesarias para refitar el scaler.")
             st.stop()
    else:
        st.error(f"❌ Error: Scaler no encontrado y '{os.path.basename(PREPROCESSED_DATA_PATH)}' tampoco encontrado para refitarlo.")
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

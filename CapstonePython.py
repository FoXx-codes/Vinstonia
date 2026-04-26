import streamlit as st
import pandas as pd
import joblib
import os
import urllib.request
import numpy as np

# --- 1. CONFIGURATION ---
# Your Google Drive ID for the +100MB model
BIG_MODEL_ID = '1jeeEGr8VTP4r7fnk__ljE9rTLKtARWAO'
BIG_MODEL_URL = f'https://drive.google.com/uc?id={BIG_MODEL_ID}'

# File names
MODEL_FILE = 'car_price_model.pkl'
MEANS_FILE = 'model_means.pkl'
COLUMNS_FILE = 'model_columns.pkl'


@st.cache_resource
def load_oracle_assets():
    """Fetches the heavy model from cloud and loads local helpers."""
    if not os.path.exists(MODEL_FILE):
        with st.spinner("Oracle is fetching the core model (100MB+)..."):
            try:
                # Direct download link for Google Drive
                urllib.request.urlretrieve(BIG_MODEL_URL, MODEL_FILE)
            except Exception as e:
                st.error(f"Cloud fetch failed: {e}")
                return None, None, None

    try:
        model = joblib.load(MODEL_FILE)
        means = joblib.load(MEANS_FILE)
        cols = joblib.load(COLUMNS_FILE)
        return model, means, cols
    except Exception as e:
        st.error(f"Error loading local files: {e}")
        return None, None, None


# --- 2. INITIALIZATION ---
model, model_means, model_columns = load_oracle_assets()

# --- 3. UI SETUP ---
st.set_page_config(page_title="Vinstonia Oracle", page_icon="🚗")
st.title("🚗 Vinstonia: Used Car Price Oracle")
st.markdown("---")

if model is None:
    st.warning(
        "The Oracle is synchronizing artifacts. Please refresh in a moment.")
    st.stop()

# --- 4. USER INPUTS (Sidebar) ---
st.sidebar.header("Vehicle Specifications")
year = st.sidebar.number_input("Year of Manufacture", 2000, 2024, 2018)
mileage = st.sidebar.number_input("Total Mileage (km)", 0, 500000, 45000)
engine_size = st.sidebar.slider("Engine Capacity (L)", 0.8, 6.0, 2.0)

# --- 5. PREDICTION LOGIC ---
if st.button("🔮 Consult the Oracle"):
    try:
        # Create input template from saved columns
        input_df = pd.DataFrame(
            np.zeros((1, len(model_columns))), columns=model_columns)

        # Map user inputs
        if 'Year' in input_df.columns:
            input_df['Year'] = year
        if 'Mileage' in input_df.columns:
            input_df['Mileage'] = mileage
        if 'Engine_Size' in input_df.columns:
            input_df['Engine_Size'] = engine_size

        # Fill missing columns with pre-calculated means
        input_df.update(pd.DataFrame([model_means], columns=model_columns))

        # Predict
        prediction = model.predict(input_df)

        st.balloons()
        st.markdown("### Oracle's Result:")
        st.success(f"The estimated market value is: **${prediction[0]:,.2f}**")

    except Exception as e:
        st.error(f"Prediction Error: {e}")

st.markdown("---")
st.caption("Capstone Project | Vinstonia Predictive Analytics")

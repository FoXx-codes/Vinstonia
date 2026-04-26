import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import gdown

# --- PAGE CONFIG (must be first Streamlit command) ---
st.set_page_config(page_title="Vinstonia Oracle",
                   page_icon="🚗", layout="centered")

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

HF_REPO = 'FoXx-codes/Vinstonia'
MODEL_FILE = os.path.join(BASE_DIR, 'car_price_model.pkl')
MEANS_FILE = os.path.join(BASE_DIR, 'model_means.pkl')
COLUMNS_FILE = os.path.join(BASE_DIR, 'model_columns.pkl')


@st.cache_resource
def load_oracle_assets():
    if not os.path.exists(MODEL_FILE):
        with st.spinner("Fetching model from cloud (this may take a moment)..."):
            try:
                from huggingface_hub import hf_hub_download
                hf_hub_download(
                    repo_id=HF_REPO,
                    filename='car_price_model.pkl',
                    local_dir=BASE_DIR
                )
            except Exception as e:
                st.error(f"Download failed: {e}")
                return None, None, None
    try:
        st.write(f"Loading model from: {MODEL_FILE}")
        st.write(f"File exists: {os.path.exists(MODEL_FILE)}")
        st.write(
            f"File size: {os.path.getsize(MODEL_FILE) if os.path.exists(MODEL_FILE) else 'N/A'}")
        model = joblib.load(MODEL_FILE)
        model_means = joblib.load(MEANS_FILE)
        model_cols = joblib.load(COLUMNS_FILE)
        return model, model_means, model_cols
    except Exception as e:
        st.error(f"Error loading local files: {e}")
        return None, None, None


model, model_means, model_columns = load_oracle_assets()

# --- UI ---
st.title("🚗 Vinstonia: Used Car Price Oracle")
st.markdown("Enter your car's details below to get an estimated market value.")
st.markdown("---")

if model is None:
    st.warning("The Oracle is synchronizing. Please refresh in a moment.")
    st.stop()

# --- VALID OPTIONS (derived from notebook) ---
BRANDS = [
    'Maruti Suzuki', 'Hyundai', 'Honda', 'Tata', 'Toyota', 'Ford',
    'Volkswagen', 'Mahindra', 'Renault', 'Kia', 'Skoda', 'Nissan',
    'Bmw', 'Mercedes-Benz', 'Datsun', 'Chevrolet', 'Fiat', 'Jeep',
    'Mg', 'Mitsubishi', 'Mini', 'Jaguar', 'Land Rover', 'Volvo',
    'Other_Rare', 'Ultra_Luxury', 'Unknown'
]

TRANSMISSIONS = ['Automatic', 'Manual', 'Unknown']
FUEL_TYPES = ['Petrol', 'Diesel', 'CNG', 'Hybrid', 'Electric', 'Unknown']
OWNER_TYPES = ['First', 'Second', 'Third', 'Unknown']
CAR_MODELS = sorted(model_means.index.tolist())

# --- INPUTS ---
col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", BRANDS)
    transmission = st.selectbox("Transmission", TRANSMISSIONS)
    fuel_type = st.selectbox("Fuel Type", FUEL_TYPES)

with col2:
    owner = st.selectbox("Owner Type", OWNER_TYPES)
    age = st.number_input("Car Age (years)", min_value=0,
                          max_value=40, value=5)
    km_driven = st.number_input(
        "Kilometres Driven", min_value=0, max_value=200000, value=50000, step=1000)

car_model = st.selectbox("Car Model", ["Unknown"] + CAR_MODELS)
posted_month = st.slider("Month Listed (1=Jan, 12=Dec)",
                         min_value=1, max_value=12, value=6)

st.markdown("---")

# --- PREDICTION ---
if st.button("🔮 Consult the Oracle", use_container_width=True):

    # 1. model_encoded — target encode using saved means, fallback to global mean
    if car_model in model_means.index:
        model_encoded = model_means[car_model]
    else:
        model_encoded = model_means.mean()

    # 2. Build a zero-filled row with all training columns
    input_df = pd.DataFrame([{col: 0 for col in model_columns}])

    # 3. Fill numeric features
    input_df['Age'] = age
    input_df['kmDriven'] = km_driven
    input_df['posted_month'] = posted_month
    input_df['model_encoded'] = model_encoded

    # 4. Transmission (drop_first=True dropped 'Automatic' — it's the baseline)
    if transmission == 'Manual' and 'Transmission_Manual' in input_df.columns:
        input_df['Transmission_Manual'] = 1
    if transmission == 'Unknown' and 'Transmission_Unknown' in input_df.columns:
        input_df['Transmission_Unknown'] = 1

    # 5. Owner (drop_first=True dropped 'First' — it's the baseline)
    if owner == 'Second' and 'Owner_Second' in input_df.columns:
        input_df['Owner_Second'] = 1
    if owner == 'Unknown' and 'Owner_Unknown' in input_df.columns:
        input_df['Owner_Unknown'] = 1

    # 6. FuelType (drop_first=True dropped 'CNG' — it's the baseline)
    fuel_col_map = {
        'Hybrid':     'FuelType_Hybrid',
        'Hybrid/Cng': 'FuelType_Hybrid/Cng',
        'Petrol':     'FuelType_Petrol',
        'Unknown':    'FuelType_Unknown',
    }
    fuel_col = fuel_col_map.get(fuel_type)
    if fuel_col and fuel_col in input_df.columns:
        input_df[fuel_col] = 1

    # 7. Brand_Grouped (drop_first=True dropped 'Audi' — it's the baseline)
    brand_col = f'Brand_Grouped_{brand}'
    if brand_col in input_df.columns:
        input_df[brand_col] = 1

    # 8. Predict (model was trained on Log_AskPrice — reverse with exp)
    log_pred = model.predict(input_df)[0]
    price_inr = np.expm1(log_pred)
    price_zar = price_inr * 0.21   # approximate INR → ZAR conversion

    st.balloons()
    st.markdown("### 🏷️ Oracle's Estimate")
    st.success(
        f"Estimated Market Value: **₹{price_inr:,.0f}** &nbsp;|&nbsp; ≈ **R {price_zar:,.0f}**")
    st.caption("Prediction based on Age, Kilometres Driven, Brand, Model, Transmission, Fuel Type, Owner history, and listing month.")

st.markdown("---")
st.caption(
    "Capstone Project · Vinstonia Predictive Analytics · Sibongile Ntsibande · 2026")

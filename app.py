import streamlit as st
import numpy as np
import json
import os

@st.cache_data
def load_model():
    if not os.path.exists("model_artifacts.json"):
        st.error("Model artifacts not found. Please run the export script first.")
        st.stop()
    with open("model_artifacts.json", "r") as f:
        return json.load(f)

artifacts = load_model()
w = np.array(artifacts["weights"]).reshape(-1, 1)
b = artifacts["bias"]
x_min = np.array(artifacts["x_min"])
x_max = np.array(artifacts["x_max"])
y_min = artifacts["y_min"]
y_max = artifacts["y_max"]

st.title("Real Estate Price Predictor")
st.markdown("Enter property details to estimate the market value.")

col1, col2 = st.columns(2)

with col1:
    sqft = st.number_input("Square Footage", min_value=500, max_value=10000, value=2500)
    bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.number_input("Number of Bathrooms", min_value=1.0, max_value=5.0, value=2.0, step=0.5)
    year_built = st.number_input("Year Built", min_value=1800, max_value=2024, value=2000)

with col2:
    lot_size = st.number_input("Lot Size (Acres)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    garage_size = st.number_input("Garage Size (Cars)", min_value=0, max_value=5, value=2)
    neighborhood_quality = st.slider("Neighborhood Quality (1-10)", min_value=1, max_value=10, value=5)

if st.button("Predict Price", type="primary"):
    engineered_year = 2023 - year_built
    raw_features = np.array([sqft, bedrooms, bathrooms, engineered_year, lot_size, garage_size, neighborhood_quality])
    scale_range = np.where((x_max - x_min) == 0, 1e-8, (x_max - x_min))
    scaled_features = (raw_features - x_min) / scale_range
    y_scaled_pred = np.dot(scaled_features, w) + b
    predicted_price = y_scaled_pred[0] * (y_max - y_min) + y_min
    st.success(f"Estimated Property Value: ${predicted_price:,.2f}")

import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('concrete_strength_model.pkl')
st.title("Concrete Strength Prediction App")
st.write("Enter the concrete mix details below to predict the compressive strength (MPa).")

# User inputs
cement = st.number_input("Cement (kg/m³)", min_value=0.0)
slag = st.number_input("Blast Furnace Slag (kg/m³)", min_value=0.0)
flyash = st.number_input("Fly Ash (kg/m³)", min_value=0.0)
water = st.number_input("Water (kg/m³)", min_value=0.0)
superplasticizer = st.number_input("Superplasticizer (kg/m³)", min_value=0.0)
coarseagg = st.number_input("Coarse Aggregate (kg/m³)", min_value=0.0)
fineagg = st.number_input("Fine Aggregate (kg/m³)", min_value=0.0)
age = st.number_input("Age (days)", min_value=0.0)

# Prediction Button
if st.button("Predict Strength"):
    inputs = np.array([[cement, slag, flyash, water, superplasticizer, coarseagg, fineagg, age]])
    strength = model.predict(inputs)
    st.success(f"Predicted Concrete Strength: {strength[0]:.2f} MPa")

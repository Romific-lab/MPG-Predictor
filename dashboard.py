import streamlit as st
import joblib
import numpy as np
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import math

# Load pretrained model
loaded_model = XGBRegressor()
loaded_model.load_model("mpg_predictor.json")

st.title("Car Efficiency Impact Predictor Dashboard")

# Sidebar input sliders
st.sidebar.header("Input Vehicle Specs")

# Cylinders
cylinders = st.sidebar.slider("Cylinders", 3, 12, step=1)

# Displacement input (L) – convert back to CUI for model
min_disp_l = round(50 * 0.0163871, 2)
max_disp_l = round(500 * 0.0163871, 2)
displacement_l = st.sidebar.slider("Displacement (L)", min_disp_l, max_disp_l, step=0.1)
displacement_cui = displacement_l / 0.0163871

# Horsepower
horsepower = st.sidebar.slider("Horsepower (HP)", 50, 250)

# Weight input (kg), convert back to lbs for model
min_weight_kg = int(1500 * 0.453592)
max_weight_kg = int(5500 * 0.453592)
weight_kg = st.sidebar.slider("Weight (kg)", min_weight_kg, max_weight_kg, step=10)
weight_lbs = weight_kg * 2.20462

# Acceleration
acceleration = st.sidebar.slider("Acceleration (0-60 time in seconds)", 5.0, 20.0)

# Model year (last two digits of year in dataset)
model_year = st.sidebar.slider("Model Year (e.g. 70 = 1970)", 70, 82)

# Origin
origin = st.sidebar.selectbox("Origin", options=[1, 2, 3], format_func=lambda x: {1: "USA", 2: "Europe", 3: "Asia"}[x])

# Degradation + projection settings
degradation_factor = st.sidebar.slider("Degradation Level (0 = perfect)", 0.0, 1.0, step=0.01)
projected_year = st.sidebar.slider("Projected Year", model_year + 1900, 2025, step=1)
k = degradation_factor * 0.06

# Predict MPG
features = np.array([[cylinders, displacement_cui, horsepower, weight_lbs, acceleration, model_year, origin]])
mpg = loaded_model.predict(features)[0]

fuel_consumption = 235.214583 / mpg  # L/100km
# Convert to fuel consumption
st.metric(label="Predicted Fuel Consumption (L/100km)", value=f"{fuel_consumption:.2f}")

# Estimate CO2 emissions (kg/L)
co2_emission = (fuel_consumption * 2.31) / 100
st.metric(label="Predicted CO2 Released (kg/L)", value=f"{co2_emission:.2f}")

def mpg_decay(mpg0, k, years):
    return [mpg0 * np.exp(-k * (year - 1982)) for year in years]

# Project over years
years = list(range(1982, projected_year + 1))

mpg_per_year = [
    mpg * np.exp(-k * (year - 1982)) / (1 + degradation_factor * math.log1p(year - 1982))
    for year in years
]


fuel_yearly = [
    235.214583 / mpg_t for mpg_t in mpg_per_year
]

co2_yearly = [
    fuel * 2.31 / 100 for fuel in fuel_yearly
]

# DataFrames
df_fuel = pd.DataFrame({
    "Year": years,
    "Fuel Consumption (L/100km)": fuel_yearly
}).set_index("Year")

df_co2 = pd.DataFrame({
    "Year": years,
    "CO2 Emissions (kg/L)": co2_yearly
}).set_index("Year")

# Plots
st.subheader("Projected Fuel Consumption Over Time")
st.line_chart(df_fuel)

st.subheader("Projected CO2 Emissions Over Time")
st.line_chart(df_co2)

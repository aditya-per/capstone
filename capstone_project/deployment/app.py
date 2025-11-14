import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download the saved model from the Hugging Face model hub
model_path = hf_hub_download(repo_id="adityasharma0511/predictive-maintenance-model", filename="best_predict_model.joblib")

# Load the saved model from the Hugging Face model hub
model = joblib.load(model_path)

# Streamlit UI for Customer Churn Prediction
st.title("Engine Predictive Maintenance App")
st.write("Engine Predictive Maintenance App is a tool to predicts whether an engine will fail or not based on the engine health parameters.")
st.write("Please enter the Enging parameters.")

# Get the inputs and save them into a dataframe
Engine_rpm = st.number_input("Engine rpms", min_value=0, max_value=5000, value=500)
Lub_oil_pressure = st.number_input("Lub oil pressure(in kPa)", min_value=0.0, max_value=20.0, value=3.0,format="%.4f")
Fuel_pressure = st.number_input("Fuel pressure (in kPa)", min_value=0.0, max_value=20.0, value=3.0,format="%.4f")
Coolant_pressure = st.number_input("Coolant pressure (in kPa)", min_value=0.0, max_value=20.0, value=1.0,format="%.4f")
lub_oil_temp = st.number_input("Lub oil temprature (in °C)", min_value=0.0, max_value=100.0, value=70.0,format="%.4f")
Coolant_temp = st.number_input("Coolant temprature (in °C)", min_value=0.0, max_value=250.0, value=100.0,format="%.4f")

# Save the inputs into a Dataframe. Convert categorical inputs to match model training
input_data = pd.DataFrame([{
    'Engine rpm': Engine_rpm,
    'Lub oil pressure': Lub_oil_pressure,
    'Fuel pressure': Fuel_pressure,
    'Coolant pressure': Coolant_pressure,
    'lub oil temp': lub_oil_temp,
    'Coolant temp': Coolant_temp
}])

# Set the classification threshold
classification_threshold = 0.425

# Predict button
if st.button("Predict"):
    prediction_proba = model.predict_proba(input_data)[0, 1]
    prediction = (prediction_proba >= classification_threshold).astype(int)
    result = "Fali" if prediction == 1 else "Not Fail"
    st.write(f"Based on the information provided, the engine is likely to {result}.")

import os
import streamlit as st
import requests

# Define the FastAPI endpoint (default: fallback for local development without Docker)
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000/predict/")

# Streamlit interface
st.title("ML Model Prediction Interface")

# Input fields for the data
data_input = st.text_input("Enter data (comma-separated)", "1, 2, 3, 4.6")

# Convert input data to list of floats
data = list(map(float, data_input.split(",")))

# Button to make prediction
if st.button("Predict"):
    response = requests.post(FASTAPI_URL, json={"data": data}, timeout=5)
    if response.status_code == 200:
        prediction = response.json().get("prediction")
        st.success(f"Prediction: {prediction}")
    else:
        st.error(f"Error in prediction: {response.status_code}")

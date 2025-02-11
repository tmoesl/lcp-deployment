from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load model
model = joblib.load("/app/model/model.pkl")  # model = joblib.load("../model/model.pkl")

# Initialize FastAPI app
app = FastAPI()


# Define Pydantic model for request body
class PredictRequest(BaseModel):
    data: list[float]


@app.get("/")
def home():
    return {"message": "ML Model API is running"}


@app.post("/predict/")
def predict(request: PredictRequest):
    input_data = np.array(request.data).reshape(1, -1)
    prediction = model.predict(input_data)
    return {"prediction": float(prediction[0])}

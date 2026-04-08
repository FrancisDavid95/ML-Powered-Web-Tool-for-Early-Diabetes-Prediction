from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np

model = joblib.load("model.pkl")

class InputData(BaseModel):
    pregnancies: int
    glucose: float
    bloodpressure: float
    skinthickness: float
    insulin: float
    bmi: float
    dpf: float
    age: int

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Diabetes AI API Running"}

@app.post("/predict")
def predict(data: InputData):
    features = np.array([[data.pregnancies, data.glucose, data.bloodpressure,
                          data.skinthickness, data.insulin, data.bmi, data.dpf, data.age]])

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1] * 100

    if probability < 30:
        risk = "Low Risk"
        advice = "Maintain healthy lifestyle"
    elif probability < 70:
        risk = "Medium Risk"
        advice = "Monitor diet and exercise"
    else:
        risk = "High Risk"
        advice = "Consult a doctor immediately"

    return {
        "prediction": int(prediction),
        "probability": round(probability, 2),
        "risk_level": risk,
        "advice": advice
    }

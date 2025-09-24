from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("hybrid_model.pkl")

@app.post("/predict")
def predict(features: list):
    data = np.array([features])
    pred = model.predict(data)[0]
    return {"prediction": int(pred)}

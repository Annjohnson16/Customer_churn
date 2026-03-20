from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("models/churn_model.pkl")
columns = joblib.load("models/columns.pkl")


@app.post("/predict")

def predict(data: dict):

    df = pd.DataFrame([data])

    df = pd.get_dummies(df)

    df = df.reindex(columns=columns, fill_value=0)

    prediction = model.predict(df)[0]

    probability = model.predict_proba(df)[0][1]

    if probability < 0.3:
        risk = "Low"
    elif probability < 0.7:
        risk = "Medium"
    else:
        risk = "High"

    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "risk": risk
    }
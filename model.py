import joblib
import pandas as pd
from config import MODEL_PATH
import logging

logging.info("Loading model...")
model = joblib.load(MODEL_PATH)
logging.info("Model loaded successfully")

def predict_survival(data: dict):
    df = pd.DataFrame([data])
    df['sex'] = df['sex'].map({'male': 0, 'female': 1})
    prediction = model.predict(df)[0]
    return "survived" if prediction == 1 else "not survived"

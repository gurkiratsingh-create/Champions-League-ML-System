import joblib
import pandas as pd

# Load model once (important for API performance)
model = joblib.load("model/xgb_model.pkl")


def predict_match(features: dict):
    df = pd.DataFrame([features])

    probabilities = model.predict_proba(df)[0]

    return {
        "home_win_prob": float(probabilities[0]),
        "draw_prob": float(probabilities[1]),
        "away_win_prob": float(probabilities[2])
    }

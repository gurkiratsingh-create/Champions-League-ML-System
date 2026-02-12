from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from model.predict import predict_match

app = FastAPI(title="Champions League Prediction API")

# Load team stats once
team_stats = pd.read_csv("data/processed/team_latest_stats.csv")


class MatchTeams(BaseModel):
    home_team: str
    away_team: str


@app.get("/")
def home():
    return {"message": "Champions League Prediction API is running."}


@app.post("/predict")
def predict(match: MatchTeams):

    home_data = team_stats[team_stats["Team"] == match.home_team]
    away_data = team_stats[team_stats["Team"] == match.away_team]

    if home_data.empty or away_data.empty:
        return {"error": "One or both teams not found."}

    home_data = home_data.iloc[0]
    away_data = away_data.iloc[0]

    features = {
        "EloDiff": home_data["Elo"] - away_data["Elo"],
        "HomeAvgGoals5": home_data["AvgGoals5"],
        "HomeAvgConceded5": home_data["AvgConceded5"],
        "HomeWinRate5": home_data["WinRate5"],
        "AwayAvgGoals5": away_data["AvgGoals5"],
        "AwayAvgConceded5": away_data["AvgConceded5"],
        "AwayWinRate5": away_data["WinRate5"]
    }

    result = predict_match(features)

    return result

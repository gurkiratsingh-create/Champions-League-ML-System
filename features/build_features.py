import pandas as pd
import os

def build_features():
    df = pd.read_csv("data/processed/matches_cleaned.csv")

    # Creating Elo Difference (VERY IMPORTANT FEATURE)
    df["EloDiff"] = df["HomeElo"] - df["AwayElo"]
    df["EloRatio"] = df["HomeElo"] / df["AwayElo"]

    # Selecting only features allowed at prediction time
    feature_columns = [
    "MatchDate",
    "EloDiff",
    "EloRatio",
    "Form5Home",
    "Form5Away",
    "Target"
]

    df = df[feature_columns]

    # Sorting by date (critical for time-based split)
    df["MatchDate"] = pd.to_datetime(df["MatchDate"])
    df = df.sort_values("MatchDate")

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/training_dataset.csv", index=False)

    print("Feature dataset saved successfully.")


if __name__ == "__main__":
    build_features()

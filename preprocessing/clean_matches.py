import pandas as pd
import os

def clean_dataset():
    df=pd.read_csv("data/raw/Matches.csv")
    df = df[[
        "MatchDate",
        "HomeTeam",
        "AwayTeam",
        "HomeElo",
        "AwayElo",
        "Form5Home",
        "Form5Away",
        "FTHome",
        "FTAway",
        "FTResult"
    ]]

    
    df = df.dropna()

    
    result_map = {
        "H": 0,  # Home win
        "D": 1,  # Draw
        "A": 2   # Away win
    }

    df["Target"] = df["FTResult"].map(result_map)


    df["MatchDate"] = pd.to_datetime(df["MatchDate"])

   
    df = df.sort_values("MatchDate")

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/matches_cleaned.csv", index=False)

    print("Cleaned dataset saved.")


if __name__ == "__main__":
    clean_dataset()
import pandas as pd
import os
from collections import defaultdict, deque

def build_rolling_features():

    df = pd.read_csv("data/processed/matches_cleaned.csv")
    df["MatchDate"] = pd.to_datetime(df["MatchDate"])
    df = df.sort_values("MatchDate")

    # Structures to store past results
    team_goals_scored = defaultdict(lambda: deque(maxlen=5))
    team_goals_conceded = defaultdict(lambda: deque(maxlen=5))
    team_results = defaultdict(lambda: deque(maxlen=5))

    rolling_data = []

    for _, row in df.iterrows():

        home = row["HomeTeam"]
        away = row["AwayTeam"]

        # ---- HOME TEAM FEATURES ----
        home_avg_scored = (
            sum(team_goals_scored[home]) / len(team_goals_scored[home])
            if len(team_goals_scored[home]) > 0 else 0
        )

        home_avg_conceded = (
            sum(team_goals_conceded[home]) / len(team_goals_conceded[home])
            if len(team_goals_conceded[home]) > 0 else 0
        )

        home_win_rate = (
            sum(team_results[home]) / len(team_results[home])
            if len(team_results[home]) > 0 else 0
        )

        # ---- AWAY TEAM FEATURES ----
        away_avg_scored = (
            sum(team_goals_scored[away]) / len(team_goals_scored[away])
            if len(team_goals_scored[away]) > 0 else 0
        )

        away_avg_conceded = (
            sum(team_goals_conceded[away]) / len(team_goals_conceded[away])
            if len(team_goals_conceded[away]) > 0 else 0
        )

        away_win_rate = (
            sum(team_results[away]) / len(team_results[away])
            if len(team_results[away]) > 0 else 0
        )

        rolling_data.append({
        "MatchDate": row["MatchDate"],
        "HomeTeam": home,
        "AwayTeam": away,
        "HomeElo": row["HomeElo"],
        "AwayElo": row["AwayElo"],
        "EloDiff": row["HomeElo"] - row["AwayElo"],
        "HomeAvgGoals5": home_avg_scored,
        "HomeAvgConceded5": home_avg_conceded,
        "HomeWinRate5": home_win_rate,
        "AwayAvgGoals5": away_avg_scored,
        "AwayAvgConceded5": away_avg_conceded,
        "AwayWinRate5": away_win_rate,
        "Target": row["Target"]
        })


        # ---- UPDATE HISTORY AFTER FEATURE COMPUTATION ----
        team_goals_scored[home].append(row["FTHome"])
        team_goals_conceded[home].append(row["FTAway"])

        team_goals_scored[away].append(row["FTAway"])
        team_goals_conceded[away].append(row["FTHome"])

        # Win = 1, Draw = 0.5, Loss = 0
        if row["Target"] == 0:
            team_results[home].append(1)
            team_results[away].append(0)
        elif row["Target"] == 2:
            team_results[home].append(0)
            team_results[away].append(1)
        else:
            team_results[home].append(0.5)
            team_results[away].append(0.5)

    rolling_df = pd.DataFrame(rolling_data)

    os.makedirs("data/processed", exist_ok=True)
    rolling_df.to_csv("data/processed/training_with_rolling.csv", index=False)

    print("Rolling feature dataset created successfully.")

if __name__ == "__main__":
    build_rolling_features()
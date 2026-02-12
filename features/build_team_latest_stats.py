import pandas as pd
import os

def build_team_latest_stats():

    df = pd.read_csv("data/processed/training_with_rolling.csv")
    df["MatchDate"] = pd.to_datetime(df["MatchDate"])
    df = df.sort_values("MatchDate")

    elo_df = pd.read_csv("data/raw/EloRatings.csv")

    teams = {}

    for _, row in df.iterrows():

        teams[row["HomeTeam"]] = {
            "Elo": row["HomeElo"],
            "AvgGoals5": row["HomeAvgGoals5"],
            "AvgConceded5": row["HomeAvgConceded5"],
            "WinRate5": row["HomeWinRate5"]
        }

        teams[row["AwayTeam"]] = {
            "Elo": row["AwayElo"],
            "AvgGoals5": row["AwayAvgGoals5"],
            "AvgConceded5": row["AwayAvgConceded5"],
            "WinRate5": row["AwayWinRate5"]
        }

    team_df = pd.DataFrame.from_dict(teams, orient="index").reset_index()
    team_df.rename(columns={"index": "Team"}, inplace=True)

    # Merge country information
    latest_country = elo_df.sort_values("date").groupby("club").last().reset_index()

    team_df = team_df.merge(
        latest_country[["club", "country"]],
        left_on="Team",
        right_on="club",
        how="left"
    )

    team_df.drop(columns=["club"], inplace=True)

    os.makedirs("data/processed", exist_ok=True)
    team_df.to_csv("data/processed/team_latest_stats.csv", index=False)

    print("Team latest stats with country created.")

if __name__ == "__main__":
    build_team_latest_stats()

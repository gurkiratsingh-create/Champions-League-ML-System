import pandas as pd
import numpy as np
from model.predict import predict_match

# Load team stats
team_stats = pd.read_csv("data/processed/team_latest_stats.csv")


def get_team_features(home_team, away_team):
    home = team_stats[team_stats["Team"] == home_team].iloc[0]
    away = team_stats[team_stats["Team"] == away_team].iloc[0]

    return {
        "EloDiff": home["Elo"] - away["Elo"],
        "HomeAvgGoals5": home["AvgGoals5"],
        "HomeAvgConceded5": home["AvgConceded5"],
        "HomeWinRate5": home["WinRate5"],
        "AwayAvgGoals5": away["AvgGoals5"],
        "AwayAvgConceded5": away["AvgConceded5"],
        "AwayWinRate5": away["WinRate5"]
    }


def simulate_match(team1, team2):
    features = get_team_features(team1, team2)
    probs = predict_match(features)

    p_home = probs["home_win_prob"]
    p_draw = probs["draw_prob"]
    p_away = probs["away_win_prob"]

    total = p_home + p_draw + p_away    
    p_home /= total
    p_draw /= total
    p_away /= total

    outcome = np.random.choice(
        ["home", "draw", "away"],
        p=[p_home, p_draw, p_away]
    )


    if outcome == "home":
        return team1
    elif outcome == "away":
        return team2
    else:
        # In knockout, resolve draw randomly
        return np.random.choice([team1, team2])


def simulate_tournament(teams):
    current_round = teams.copy()

    while len(current_round) > 1:
        np.random.shuffle(current_round)
        next_round = []

        for i in range(0, len(current_round), 2):
            winner = simulate_match(current_round[i], current_round[i+1])
            next_round.append(winner)

        current_round = next_round

    return current_round[0]


def monte_carlo_simulation(teams, n_simulations=1000):

    win_counts = {team: 0 for team in teams}

    for _ in range(n_simulations):
        winner = simulate_tournament(teams)
        win_counts[winner] += 1

    results = {
        team: win_counts[team] / n_simulations
        for team in teams
    }

    return results 

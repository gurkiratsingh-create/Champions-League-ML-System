import pandas as pd
import numpy as np
from model.predict import predict_match

# Load team stats once
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


def precompute_match_probabilities(teams):
    """
    Precompute probabilities for all unique matchups.
    This avoids calling the ML model thousands of times.
    """
    prob_cache = {}

    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            team1 = teams[i]
            team2 = teams[j]

            features = get_team_features(team1, team2)
            probs = predict_match(features)

            # Normalize just in case
            total = probs["home_win_prob"] + probs["draw_prob"] + probs["away_win_prob"]
            probs["home_win_prob"] /= total
            probs["draw_prob"] /= total
            probs["away_win_prob"] /= total

            prob_cache[(team1, team2)] = probs
            prob_cache[(team2, team1)] = {
                "home_win_prob": probs["away_win_prob"],
                "draw_prob": probs["draw_prob"],
                "away_win_prob": probs["home_win_prob"]
            }

    return prob_cache


def simulate_match(team1, team2, prob_cache):
    probs = prob_cache[(team1, team2)]

    outcome = np.random.choice(
        ["home", "draw", "away"],
        p=[
            probs["home_win_prob"],
            probs["draw_prob"],
            probs["away_win_prob"]
        ]
    )

    if outcome == "home":
        return team1
    elif outcome == "away":
        return team2
    else:
        return np.random.choice([team1, team2])


def simulate_tournament(teams, prob_cache):
    current_round = teams.copy()

    while len(current_round) > 1:
        np.random.shuffle(current_round)
        next_round = []

        for i in range(0, len(current_round), 2):
            winner = simulate_match(
                current_round[i],
                current_round[i + 1],
                prob_cache
            )
            next_round.append(winner)

        current_round = next_round

    return current_round[0]


def monte_carlo_simulation(teams, n_simulations=1000):

    win_counts = {team: 0 for team in teams}

    # 🔥 Precompute once
    prob_cache = precompute_match_probabilities(teams)

    for _ in range(n_simulations):
        winner = simulate_tournament(teams, prob_cache)
        win_counts[winner] += 1

    return {
        team: win_counts[team] / n_simulations
        for team in teams
    }

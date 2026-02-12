from model.predict import predict_match

sample_input = {
    "EloDiff": 50,
    "HomeAvgGoals5": 1.8,
    "HomeAvgConceded5": 1.0,
    "HomeWinRate5": 0.6,
    "AwayAvgGoals5": 1.2,
    "AwayAvgConceded5": 1.4,
    "AwayWinRate5": 0.4
}

result = predict_match(sample_input)

print(result)

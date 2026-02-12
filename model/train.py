from tkinter.filedialog import test
import pandas as pd
from xgboost import XGBClassifier, train
from sklearn.metrics import log_loss, accuracy_score
import joblib
import os

def train_model():
    df = pd.read_csv("data/processed/training_with_rolling.csv")

    df["MatchDate"] = pd.to_datetime(df["MatchDate"])

    # Time-based split
    split_date = "2018-01-01"

    train = df[df["MatchDate"] < split_date]
    test = df[df["MatchDate"] >= split_date]

    X_train = train[[
    "EloDiff",
    "HomeAvgGoals5",
    "HomeAvgConceded5",
    "HomeWinRate5",
    "AwayAvgGoals5",
    "AwayAvgConceded5",
    "AwayWinRate5"
    ]]

    y_train = train["Target"]

    X_test = test[[
        "EloDiff",
        "HomeAvgGoals5",
        "HomeAvgConceded5",
        "HomeWinRate5",
        "AwayAvgGoals5",
        "AwayAvgConceded5",
        "AwayWinRate5"
    ]]

    y_test = test["Target"]

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="mlogloss"
    )

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)
    preds = model.predict(X_test)

    print("Log Loss:", log_loss(y_test, probs))
    print("Accuracy:", accuracy_score(y_test, preds))

    # Save model
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/xgb_model.pkl")

    print("Model saved successfully.")

if __name__ == "__main__":
    train_model()

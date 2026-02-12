# ⚽ Champions League Prediction Engine

An end-to-end Machine Learning system that predicts football match outcomes and simulates Champions League-style knockout tournaments using probabilistic modeling and Monte Carlo simulation.

---

## 🚀 Live Features

- 🔮 Match outcome prediction (Home / Draw / Away probabilities)
- 🏆 Tournament simulation (Top 16 European clubs)
- 📊 Monte Carlo simulation for title probability estimation
- 🌍 Realistic European league filtering
- 🎨 Modern interactive Streamlit dashboard
- ⚡ FastAPI backend for model inference

---

## 🧠 Project Architecture

Historical Match Data
↓
Feature Engineering (Rolling stats + Elo)
↓
Model Training (XGBoost, time-aware split)
↓
Saved Model (.pkl)
↓
FastAPI Inference API
↓
Streamlit Interactive Dashboard
↓
Monte Carlo Tournament Simulation


---

## 📊 Machine Learning Approach

### 1️⃣ Data Processing
- Chronological sorting (prevents data leakage)
- Rolling 5-match statistics:
  - Average goals scored
  - Average goals conceded
  - Win rate
- Elo-based strength differential

### 2️⃣ Model
- XGBoost multi-class classifier
- Time-based train/test split
- Probabilistic predictions
- Log loss + accuracy evaluation

### 3️⃣ Tournament Logic
- Composite Strength Score:
  - 50% Elo (normalized)
  - 20% Win rate (recent form)
  - 20% Attack strength
  - 10% Defensive strength
- Top 16 clubs from major European leagues
- Monte Carlo simulation (100–5000 runs)

---

## 🏆 Tournament Simulation

Simulates a knockout bracket:

- Round of 16
- Quarterfinals
- Semifinals
- Final

Each match outcome is sampled from predicted probabilities.
Draws are resolved randomly (knockout format).

Outputs:
- Title probability for each club
- Podium visualization (Top 3)
- Full ranking table

---

## 🛠 Tech Stack

**Machine Learning**
- XGBoost
- Scikit-learn
- Pandas
- NumPy

**Backend**
- FastAPI
- Uvicorn

**Frontend**
- Streamlit
- Plotly

**Simulation**
- Monte Carlo probability sampling

---

## 📂 Project Structure

champions_league_ml/
│
├── api/
│ └── main.py
│
├── model/
│ ├── train.py
│ ├── predict.py
│ └── xgb_model.pkl
│
├── features/
│ ├── build_features.py
│ ├── build_rolling_features.py
│ └── build_team_latest_stats.py
│
├── simulation/
│ └── simulate_tournament.py
│
├── app/
│ └── dashboard.py
│
├── data/
│ ├── raw/
│ └── processed/
│
├── requirements.txt
└── Procfile


---

## 🧪 Local Setup


### 1️⃣ Create Virtual Environment


python -m venv venv


Activate:



venv\Scripts\activate # Windows
source venv/bin/activate # Mac/Linux


### 2️⃣ Install Dependencies



pip install -r requirements.txt


### 3️⃣ Run Backend



uvicorn api.main:app --reload


### 4️⃣ Run Frontend



streamlit run app/dashboard.py


---

## 📈 Model Evaluation

- Multi-class classification (Home / Draw / Away)
- Log Loss optimization
- Time-aware validation split
- Probabilistic predictions (not hard labels)

---

## 🎯 Key Engineering Highlights

- Prevented data leakage with chronological feature computation
- Separated training and inference pipelines
- Modular architecture (API, model, simulation, UI)
- Environment variable-based API configuration
- Robust error handling
- Realistic competition filtering
- Professional dashboard design

---

## 🧑‍💻 Author

Gurkirat Singh  
Machine Learning & AI Engineering Enthusiast  

---

## 📌 Future Improvements

- SHAP explainability
- Two-leg aggregate simulation
- Expected Goals (xG) modeling
- Dockerized deployment
- Automated CI/CD pipeline

---

## ⚠ Disclaimer

This project is for educational and analytical purposes.  
Predictions are probabilistic and do not guarantee outcome.
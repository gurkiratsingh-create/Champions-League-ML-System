import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
import requests
import pandas as pd
import numpy as np
from simulation.simulate_tournament import monte_carlo_simulation
import plotly.graph_objects as go
import plotly.express as px

# ============================
# PAGE CONFIGURATION
# ============================
st.set_page_config(
    page_title="Champions League Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================
# CUSTOM CSS STYLING
# ============================
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Poppins:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
        font-family: 'Poppins', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main Title Styling */
    .main-title {
        font-family: 'Bebas Neue', cursive;
        font-size: 4.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        animation: titleGlow 3s ease-in-out infinite;
    }
    
    @keyframes titleGlow {
        0%, 100% { filter: drop-shadow(0 0 20px rgba(102, 126, 234, 0.5)); }
        50% { filter: drop-shadow(0 0 40px rgba(118, 75, 162, 0.8)); }
    }
    
    .subtitle {
        text-align: center;
        color: #a8b2d1;
        font-size: 1.2rem;
        font-weight: 300;
        margin-bottom: 3rem;
        letter-spacing: 2px;
    }
    
    /* Section Headers */
    .section-header {
        font-family: 'Bebas Neue', cursive;
        font-size: 2.5rem;
        color: #ffffff;
        margin-top: 3rem;
        margin-bottom: 1.5rem;
        padding-left: 20px;
        border-left: 5px solid #667eea;
        letter-spacing: 2px;
    }
    
    /* Card Containers */
    .prediction-card {
        background: rgba(26, 31, 58, 0.6);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(102, 126, 234, 0.2);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    
    .prediction-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 50px rgba(102, 126, 234, 0.3);
        border: 1px solid rgba(102, 126, 234, 0.4);
    }
    
    /* Probability Display */
    .prob-container {
        background: rgba(15, 20, 35, 0.8);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    .prob-label {
        font-size: 1.1rem;
        font-weight: 600;
        color: #a8b2d1;
        margin-bottom: 0.5rem;
    }
    
    .prob-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Team Selection Info */
    .team-info {
        background: rgba(102, 126, 234, 0.1);
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
    }
    
    /* Buttons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-family: 'Poppins', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.8rem 2rem;
        border: none;
        border-radius: 50px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.6);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Select Boxes */
    .stSelectbox > div > div {
        background: rgba(26, 31, 58, 0.8);
        border: 1px solid rgba(102, 126, 234, 0.3);
        border-radius: 10px;
        color: white;
    }
    
    /* Progress Bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Warning/Error Messages */
    .stAlert {
        background: rgba(255, 107, 107, 0.1);
        border: 1px solid rgba(255, 107, 107, 0.3);
        border-radius: 10px;
        color: #ff6b6b;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background: rgba(102, 126, 234, 0.2);
    }
    
    /* Metric Cards */
    .metric-card {
        background: rgba(26, 31, 58, 0.6);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(102, 126, 234, 0.2);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        border: 1px solid rgba(102, 126, 234, 0.5);
        transform: scale(1.02);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #a8b2d1;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Divider */
    .divider {
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, #667eea 50%, transparent 100%);
        margin: 3rem 0;
    }
    
    /* Animation for cards */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animated {
        animation: fadeInUp 0.6s ease-out;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(26, 31, 58, 0.5);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
</style>
""", unsafe_allow_html=True)

# ============================
# HEADER
# ============================
st.markdown('<h1 class="main-title">⚽ Champions League Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Match Predictions & Tournament Simulations</p>', unsafe_allow_html=True)

# ============================
# LOAD DATA
# ============================
@st.cache_data
def load_team_stats():
    return pd.read_csv("data/processed/team_latest_stats.csv")

team_stats = load_team_stats()
teams = sorted(team_stats["Team"].unique())

# ============================
# MATCH PREDICTION SECTION
# ============================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<h2 class="section-header">🔮 Match Prediction</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="prediction-card animated">', unsafe_allow_html=True)
    st.markdown("### 🏠 Home Team")
    home_team = st.selectbox("Select Home Team", teams, key="home_team", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="prediction-card animated">', unsafe_allow_html=True)
    st.markdown("### 🚩 Away Team")
    away_team = st.selectbox("Select Away Team", teams, key="away_team", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

if home_team == away_team:
    st.warning("⚠️ Home and Away teams must be different.")

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🎯 Predict Match Outcome") and home_team != away_team:
    
    payload = {
        "home_team": home_team,
        "away_team": away_team
    }

    try:
        with st.spinner("🔄 Analyzing match data..."):
            API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
            response = requests.post(
                    f"{API_URL}/predict",
                    json=payload,
                timeout=10)

        result = response.json()

        if "error" in result:
            st.error(f"❌ {result['error']}")
        else:
            st.markdown('<div class="prediction-card animated">', unsafe_allow_html=True)
            st.markdown("### 📊 Match Outcome Probabilities")
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Create three columns for probabilities
            prob_col1, prob_col2, prob_col3 = st.columns(3)
            
            with prob_col1:
                st.markdown('<div class="prob-container">', unsafe_allow_html=True)
                st.markdown('<div class="prob-label">🏠 Home Win</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="prob-value">{result["home_win_prob"]*100:.1f}%</div>', unsafe_allow_html=True)
                st.progress(result["home_win_prob"])
                st.markdown('</div>', unsafe_allow_html=True)
            
            with prob_col2:
                st.markdown('<div class="prob-container">', unsafe_allow_html=True)
                st.markdown('<div class="prob-label">🤝 Draw</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="prob-value">{result["draw_prob"]*100:.1f}%</div>', unsafe_allow_html=True)
                st.progress(result["draw_prob"])
                st.markdown('</div>', unsafe_allow_html=True)
            
            with prob_col3:
                st.markdown('<div class="prob-container">', unsafe_allow_html=True)
                st.markdown('<div class="prob-label">🚩 Away Win</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="prob-value">{result["away_win_prob"]*100:.1f}%</div>', unsafe_allow_html=True)
                st.progress(result["away_win_prob"])
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Create interactive visualization
            fig = go.Figure(data=[
                go.Bar(
                    x=['Home Win', 'Draw', 'Away Win'],
                    y=[result["home_win_prob"]*100, result["draw_prob"]*100, result["away_win_prob"]*100],
                    marker=dict(
                        color=['#667eea', '#a8b2d1', '#764ba2'],
                        line=dict(color='rgba(102, 126, 234, 0.5)', width=2)
                    ),
                    text=[f'{result["home_win_prob"]*100:.1f}%', 
                          f'{result["draw_prob"]*100:.1f}%', 
                          f'{result["away_win_prob"]*100:.1f}%'],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title=f"{home_team} vs {away_team}",
                xaxis_title="Outcome",
                yaxis_title="Probability (%)",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#a8b2d1', family='Poppins'),
                title_font=dict(size=20, color='#ffffff'),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

    except requests.exceptions.Timeout:
        st.error("⏱️ Request timed out. Please try again.")
    except requests.exceptions.ConnectionError:
        st.error("🔌 Could not connect to API. Make sure FastAPI is running on http://127.0.0.1:8000")
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")

# ============================
# TOURNAMENT SIMULATION SECTION
# ============================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown('<h2 class="section-header">🏆 Tournament Simulation</h2>', unsafe_allow_html=True)

st.markdown('<div class="prediction-card animated">', unsafe_allow_html=True)
st.markdown("### 🌟 Elite European Tournament (Top 16 Clubs)")
st.markdown("Simulating a knockout tournament with the strongest teams from major European leagues")
st.markdown('</div>', unsafe_allow_html=True)

# Copy team stats
ranking_df = team_stats.copy()

# Filter only major European league teams
major_divisions = ["E0", "SP1", "D1", "I1", "F1", "P1", "N1", "B1"]

@st.cache_data
def load_matches():
    return pd.read_csv("data/raw/Matches.csv")

matches_df = load_matches()

european_teams = matches_df.query(
    "Division in @major_divisions"
)["HomeTeam"].unique()

ranking_df = ranking_df[ranking_df["Team"].isin(european_teams)]

# Normalize metrics
ranking_df["Elo_norm"] = (
    (ranking_df["Elo"] - ranking_df["Elo"].min()) /
    (ranking_df["Elo"].max() - ranking_df["Elo"].min())
)

ranking_df["Goals_norm"] = (
    (ranking_df["AvgGoals5"] - ranking_df["AvgGoals5"].min()) /
    (ranking_df["AvgGoals5"].max() - ranking_df["AvgGoals5"].min())
)

ranking_df["DefenseScore"] = 1 / (ranking_df["AvgConceded5"] + 0.01)

ranking_df["Defense_norm"] = (
    (ranking_df["DefenseScore"] - ranking_df["DefenseScore"].min()) /
    (ranking_df["DefenseScore"].max() - ranking_df["DefenseScore"].min())
)

# Composite Strength Score
ranking_df["StrengthScore"] = (
    0.5 * ranking_df["Elo_norm"] +
    0.2 * ranking_df["WinRate5"] +
    0.2 * ranking_df["Goals_norm"] +
    0.1 * ranking_df["Defense_norm"]
)

# Select top 16
top_16 = (
    ranking_df.sort_values("StrengthScore", ascending=False)
    .head(16)["Team"]
    .tolist()
)

if len(top_16) < 16:
    st.error("❌ Not enough European teams found in the dataset.")
else:
    st.markdown('<div class="prediction-card animated">', unsafe_allow_html=True)
    st.markdown("### 🎖️ Qualified Teams (by Composite Strength)")
    
    # Display teams in a grid
    cols = st.columns(4)
    for idx, team in enumerate(top_16):
        with cols[idx % 4]:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">#{idx + 1}</div>
                <div style="font-size: 1.2rem; font-weight: 600; color: white; margin-top: 0.5rem;">
                    {team}
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Simulation controls
    st.markdown('<div class="prediction-card animated">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Simulation Settings")
    
    num_simulations = st.slider(
        "Number of Monte Carlo Simulations",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
        key="top16_sim_slider",
        help="More simulations provide more accurate probability estimates"
    )
    
    st.markdown(f"""
    <div class="team-info">
        <strong>ℹ️ Simulation Info:</strong> Running {num_simulations:,} simulations to calculate 
        tournament win probabilities for each team based on their strength metrics.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("🚀 Simulate Tournament"):
        
        with st.spinner(f"⚡ Running {num_simulations:,} Monte Carlo simulations..."):
            results = monte_carlo_simulation(top_16, n_simulations=num_simulations)

        result_df = pd.DataFrame(
            results.items(),
            columns=["Team", "Win Probability"]
        ).sort_values("Win Probability", ascending=False)
        
        st.markdown('<div class="prediction-card animated">', unsafe_allow_html=True)
        st.markdown("### 🏆 Tournament Win Probabilities")
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Display top 3 in special cards
        top3_cols = st.columns(3)
        medals = ["🥇", "🥈", "🥉"]
        colors = ["#FFD700", "#C0C0C0", "#CD7F32"]
        
        for idx, (col, medal, color) in enumerate(zip(top3_cols, medals, colors)):
            with col:
                team_name = result_df.iloc[idx]["Team"]
                prob = result_df.iloc[idx]["Win Probability"]
                st.markdown(f"""
                <div class="metric-card" style="border: 2px solid {color};">
                    <div style="font-size: 3rem;">{medal}</div>
                    <div style="font-size: 1.5rem; font-weight: 700; color: white; margin: 0.5rem 0;">
                        {team_name}
                    </div>
                    <div class="metric-value" style="color: {color};">
                        {prob*100:.2f}%
                    </div>
                    <div class="metric-label">Win Probability</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Interactive bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=result_df["Win Probability"] * 100,
                y=result_df["Team"],
                orientation='h',
                marker=dict(
                    color=result_df["Win Probability"] * 100,
                    colorscale='Viridis',
                    line=dict(color='rgba(102, 126, 234, 0.5)', width=1)
                ),
                text=[f'{prob*100:.2f}%' for prob in result_df["Win Probability"]],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Complete Tournament Win Probability Rankings",
            xaxis_title="Win Probability (%)",
            yaxis_title="Team",
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#a8b2d1', family='Poppins'),
            title_font=dict(size=20, color='#ffffff'),
            height=600,
            yaxis=dict(autorange="reversed")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Data table
        st.markdown("### 📋 Detailed Results")
        result_df["Win Probability (%)"] = (result_df["Win Probability"] * 100).round(2)
        result_df["Rank"] = range(1, len(result_df) + 1)
        st.dataframe(
            result_df[["Rank", "Team", "Win Probability (%)"]],
            hide_index=True,
            use_container_width=True
        )
        
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #a8b2d1; padding: 2rem; font-size: 0.9rem;">
    <p>Powered by Machine Learning & Monte Carlo Simulation</p>
    <p style="font-size: 0.8rem; margin-top: 0.5rem; opacity: 0.7;">
        Data-driven predictions for the beautiful game ⚽
    </p>
</div>
""", unsafe_allow_html=True)
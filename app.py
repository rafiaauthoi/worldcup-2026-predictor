import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
from collections import defaultdict

# Page config
st.set_page_config(
    page_title="xPredict",
    page_icon="⚽",
    layout="wide"
)

# Remove top padding
st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and data
@st.cache_resource
def load_model():
    with open('data/processed/best_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_resource
def load_elo():
    with open('data/processed/elo_ratings.json', 'r') as f:
        return json.load(f)

model = load_model()
elo_ratings = load_elo()

# World Cup 2026 teams
WC_TEAMS = sorted([
    'Argentina', 'France', 'Spain', 'England', 'Brazil', 'Portugal',
    'Germany', 'Netherlands', 'Belgium', 'Croatia', 'Uruguay', 'Mexico',
    'United States', 'Canada', 'Morocco', 'Senegal', 'Japan', 'South Korea',
    'Australia', 'Ecuador', 'Colombia', 'Chile', 'Peru', 'Venezuela',
    'Saudi Arabia', 'Iran', 'South Africa', 'Nigeria', 'Cameroon', 'Ghana',
    'New Zealand', 'Switzerland', 'Austria', 'Denmark', 'Serbia', 'Ukraine',
    'Poland', 'Czechia', 'Slovakia', 'Hungary', 'Romania', 'Slovenia',
    'Turkey', 'Albania', 'Panama', 'Costa Rica', 'Honduras', 'Jamaica'
])

# Title
st.title("xPredict | FIFA World Cup 2026")
st.markdown("Machine learning predictions powered by 49,000+ historical matches and 10,000 Monte Carlo simulations.")

# Tabs
tab1, tab2, tab3 = st.tabs(["Championship Odds", "Match Predictor", "Tournament Simulator"])

with tab1:
    st.header("Championship Odds")
    st.markdown("Based on 10,000 Monte Carlo simulations using Elo ratings from 49,000+ historical matches.")

    @st.cache_data
    def run_simulation(n=10000):
        championship_counts = defaultdict(int)
        match_cache = {}

        def get_win_prob(team1, team2):
            key = (team1, team2)
            if key not in match_cache:
                elo1 = elo_ratings.get(team1, 1000)
                elo2 = elo_ratings.get(team2, 1000)
                p1 = 1 / (1 + 10 ** ((elo2 - elo1) / 400))
                match_cache[key] = p1
                match_cache[(team2, team1)] = 1 - p1
            return match_cache[key]

        def simulate_match(team1, team2):
            p1 = get_win_prob(team1, team2)
            return team1 if np.random.random() < p1 else team2

        def simulate_tournament(teams):
            remaining = teams.copy()
            np.random.shuffle(remaining)
            while len(remaining) > 1:
                next_round = []
                for i in range(0, len(remaining), 2):
                    if i + 1 < len(remaining):
                        winner = simulate_match(remaining[i], remaining[i+1])
                        next_round.append(winner)
                    else:
                        next_round.append(remaining[i])
                remaining = next_round
            return remaining[0]

        for _ in range(n):
            champion = simulate_tournament(WC_TEAMS)
            championship_counts[champion] += 1

        probs = {team: count/n for team, count in championship_counts.items()}
        return dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))

    probs = run_simulation()

    top15 = dict(list(probs.items())[:15])
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(top15.keys(), [v*100 for v in top15.values()], color='gold', edgecolor='black')
    ax.set_title('FIFA World Cup 2026 | Championship Probability (10,000 Simulations)')
    ax.set_xlabel('Team')
    ax.set_ylabel('Probability (%)')
    plt.xticks(rotation=90, ha='right', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Full Rankings")
    df_probs = pd.DataFrame(list(probs.items()), columns=['Team', 'Probability'])
    df_probs['Probability'] = df_probs['Probability'].apply(lambda x: f"{x:.1%}")
    st.dataframe(df_probs, use_container_width=True, hide_index=True)

with tab2:
    st.header("Match Predictor")
    st.markdown("Select any two teams to get win probabilities based on our model.")

    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Home Team", WC_TEAMS, index=WC_TEAMS.index('Argentina'))
    with col2:
        team2 = st.selectbox("Away Team", WC_TEAMS, index=WC_TEAMS.index('France'))

    if team1 == team2:
        st.warning("Please select two different teams.")
    else:
        elo1 = elo_ratings.get(team1, 1000)
        elo2 = elo_ratings.get(team2, 1000)

        features = pd.DataFrame([{
            'home_elo': elo1,
            'away_elo': elo2,
            'elo_diff': elo1 - elo2,
            'home_form': 0.5,
            'away_form': 0.5,
            'form_diff': 0.0,
            'neutral': 1,
            'is_wc': 1
        }])

        proba = model.predict_proba(features)[0]
        away_win, draw, home_win = proba[0], proba[1], proba[2]

        st.subheader(f"{team1} vs {team2}")

        col1, col2, col3 = st.columns(3)
        col1.metric(f"{team1} Win", f"{home_win:.1%}")
        col2.metric("Draw", f"{draw:.1%}")
        col3.metric(f"{team2} Win", f"{away_win:.1%}")

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar([team1, 'Draw', team2], [home_win*100, draw*100, away_win*100],
               color=['steelblue', 'gray', 'salmon'])
        ax.set_ylabel('Probability (%)')
        ax.set_title(f'{team1} vs {team2} | Match Prediction')
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown(f"**Elo Ratings:** {team1}: {elo1:.0f} | {team2}: {elo2:.0f}")

with tab3:
    st.header("Tournament Simulator")
    st.markdown("Simulate the entire FIFA World Cup 2026 bracket.")

    if st.button("Run Simulation"):
        with st.spinner("Simulating 10,000 tournaments..."):
            sim_probs = run_simulation()

        st.subheader("Championship Probabilities")
        df_sim = pd.DataFrame(list(sim_probs.items()), columns=['Team', 'Probability'])
        df_sim['Probability'] = df_sim['Probability'].apply(lambda x: f"{x:.1%}")
        st.dataframe(df_sim, use_container_width=True, hide_index=True)
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
tab1, tab2, tab3, tab4 = st.tabs(["Championship Odds", "Match Predictor", "Tournament Simulator", "Player Scout"])

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
    st.markdown("Simulate the entire FIFA World Cup 2026 bracket and see who comes out on top.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Settings")
        n_sims = st.slider("Number of Simulations", min_value=1000, max_value=20000, value=10000, step=1000)
        run_btn = st.button("Run Simulation", type="primary")

    if run_btn:
        with st.spinner(f"Simulating {n_sims:,} tournaments..."):
            sim_probs = run_simulation(n_sims)

        st.subheader("Predicted Champion Probabilities")

        top10 = dict(list(sim_probs.items())[:10])
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(list(top10.keys())[::-1], [v*100 for v in list(top10.values())[::-1]],
                       color='gold', edgecolor='black')
        ax.set_xlabel('Probability (%)')
        ax.set_title(f'Top 10 | Championship Probability ({n_sims:,} Simulations)')
        for bar, val in zip(bars, list(top10.values())[::-1]):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                    f'{val:.1%}', va='center', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)

        st.subheader("Full Results")
        df_sim = pd.DataFrame(list(sim_probs.items()), columns=['Team', 'Championship Probability'])
        df_sim['Championship Probability'] = df_sim['Championship Probability'].apply(lambda x: f"{x:.1%}")
        df_sim.insert(0, 'Rank', range(1, len(df_sim) + 1))
        st.dataframe(df_sim, use_container_width=True, hide_index=True)
    else:
        st.info("Set the number of simulations and click Run Simulation to get started.")
        
with tab4:
    st.header("Player Scout")
    st.markdown("Explore player stats from the 2022 FIFA World Cup — the most recent complete tournament dataset available.")

    # Load player stats
    @st.cache_data
    def load_player_stats():
        return pd.read_csv('data/processed/player_stats_wc2022.csv')

    player_stats = load_player_stats()

    # Team selector
    teams = sorted(player_stats['team'].unique().tolist())
    selected_team = st.selectbox("Select a Team", teams, index=teams.index('Argentina'))

    team_data = player_stats[player_stats['team'] == selected_team].copy()

    # Team overview metrics
    st.subheader(f"{selected_team} | WC 2022 Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Goals", int(team_data['goals'].sum()))
    col2.metric("Total Shots", int(team_data['shots'].sum()))
    col3.metric("Total xG", f"{team_data['xg'].sum():.2f}")
    col4.metric("Yellow Cards", int(team_data['yellow_cards'].sum()))

    # Top scorers chart
    st.subheader("Top Scorers")
    top_scorers = team_data[team_data['goals'] > 0].sort_values('goals', ascending=False).head(7)

    if len(top_scorers) > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        x = range(len(top_scorers))
        bars1 = ax.bar([i - 0.2 for i in x], top_scorers['goals'], 0.4, label='Goals', color='gold', edgecolor='black')
        bars2 = ax.bar([i + 0.2 for i in x], top_scorers['xg'], 0.4, label='xG', color='steelblue', edgecolor='black')
        ax.set_xticks(list(x))
        ax.set_xticklabels(top_scorers['player'], rotation=30, ha='right', fontsize=9)
        ax.set_ylabel('Goals / xG')
        ax.set_title(f'{selected_team} | Goals vs Expected Goals (xG)')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("No goals scored by this team in WC 2022.")

    # Key players table
    st.subheader("Full Squad Stats")
    display_cols = ['player', 'goals', 'shots', 'xg', 'assists', 'passes', 'yellow_cards']
    squad_display = team_data[display_cols].sort_values('goals', ascending=False).reset_index(drop=True)
    squad_display.insert(0, 'Rank', range(1, len(squad_display) + 1))
    squad_display['xg'] = squad_display['xg'].round(2)
    st.dataframe(squad_display, use_container_width=True, hide_index=True)

    # Card risk section
    st.subheader("Card Risk Players")
    card_risk = team_data[team_data['yellow_cards'] > 0].sort_values('yellow_cards', ascending=False)
    if len(card_risk) > 0:
        st.dataframe(card_risk[['player', 'yellow_cards']].reset_index(drop=True), 
                    use_container_width=True, hide_index=True)
    else:
        st.info("No yellow cards recorded for this team.")        
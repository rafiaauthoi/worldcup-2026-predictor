import pandas as pd
import numpy as np

ELO_START = 1000
K = 30

def compute_elo_ratings(results):
    """
    Process all matches chronologically and compute Elo ratings.
    Returns the results dataframe with elo columns added,
    and the final elo ratings dictionary.
    """
    elo_ratings = {}

    def get_elo(team):
        return elo_ratings.get(team, ELO_START)

    def update_elo(winner, loser, draw=False):
        elo_w = get_elo(winner)
        elo_l = get_elo(loser)
        expected_w = 1 / (1 + 10 ** ((elo_l - elo_w) / 400))
        expected_l = 1 - expected_w
        actual_w = 0.5 if draw else 1
        actual_l = 0.5 if draw else 0
        elo_ratings[winner] = elo_w + K * (actual_w - expected_w)
        elo_ratings[loser] = elo_l + K * (actual_l - expected_l)

    home_elos, away_elos = [], []

    for _, row in results.iterrows():
        home = row['home_team']
        away = row['away_team']
        home_elos.append(get_elo(home))
        away_elos.append(get_elo(away))

        if row['home_score'] > row['away_score']:
            update_elo(home, away)
        elif row['home_score'] < row['away_score']:
            update_elo(away, home)
        else:
            update_elo(home, away, draw=True)

    results = results.copy()
    results['home_elo'] = home_elos
    results['away_elo'] = away_elos
    results['elo_diff'] = results['home_elo'] - results['away_elo']

    return results, elo_ratings


def compute_form(results, n=10):
    """
    Compute recent form (win rate over last n matches) for each team.
    """
    form = {}
    home_form, away_form = [], []

    for _, row in results.iterrows():
        home = row['home_team']
        away = row['away_team']

        home_history = form.get(home, [])
        away_history = form.get(away, [])

        home_form.append(np.mean(home_history[-n:]) if home_history else 0.5)
        away_form.append(np.mean(away_history[-n:]) if away_history else 0.5)

        if row['home_score'] > row['away_score']:
            form.setdefault(home, []).append(1)
            form.setdefault(away, []).append(0)
        elif row['home_score'] < row['away_score']:
            form.setdefault(home, []).append(0)
            form.setdefault(away, []).append(1)
        else:
            form.setdefault(home, []).append(0.5)
            form.setdefault(away, []).append(0.5)

    results = results.copy()
    results['home_form'] = home_form
    results['away_form'] = away_form
    results['form_diff'] = results['home_form'] - results['away_form']

    return results


def build_features(results):
    """
    Full feature engineering pipeline.
    Returns X (features), y (target), and final elo ratings.
    """
    results = results.dropna(subset=['home_score', 'away_score'])
    results = results.sort_values('date').reset_index(drop=True)

    results, elo_ratings = compute_elo_ratings(results)
    results = compute_form(results)

    def get_result(row):
        if row['home_score'] > row['away_score']:
            return 2
        elif row['home_score'] < row['away_score']:
            return 0
        else:
            return 1

    results['target'] = results.apply(get_result, axis=1)
    results['is_wc'] = (results['tournament'] == 'FIFA World Cup').astype(int)
    results['neutral'] = results['neutral'].astype(int)

    df = results[results['date'].dt.year >= 1990].copy()

    features = ['home_elo', 'away_elo', 'elo_diff',
                'home_form', 'away_form', 'form_diff',
                'neutral', 'is_wc']

    X = df[features]
    y = df['target']

    return X, y, elo_ratings
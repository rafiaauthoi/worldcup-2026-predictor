import numpy as np
from collections import defaultdict


def get_win_prob(team1, team2, elo_ratings):
    """
    Compute win probability for team1 vs team2 using Elo ratings.
    """
    elo1 = elo_ratings.get(team1, 1000)
    elo2 = elo_ratings.get(team2, 1000)
    return 1 / (1 + 10 ** ((elo2 - elo1) / 400))


def simulate_match(team1, team2, elo_ratings):
    """
    Simulate a single knockout match. Returns the winner.
    """
    p1 = get_win_prob(team1, team2, elo_ratings)
    return team1 if np.random.random() < p1 else team2


def simulate_tournament(teams, elo_ratings):
    """
    Simulate a single-elimination tournament.
    Returns the champion.
    """
    remaining = teams.copy()
    np.random.shuffle(remaining)

    while len(remaining) > 1:
        next_round = []
        for i in range(0, len(remaining), 2):
            if i + 1 < len(remaining):
                winner = simulate_match(remaining[i], remaining[i+1], elo_ratings)
                next_round.append(winner)
            else:
                next_round.append(remaining[i])
        remaining = next_round

    return remaining[0]


def run_simulation(teams, elo_ratings, n=10000):
    """
    Run n Monte Carlo tournament simulations.
    Returns a dict of team -> championship probability, sorted descending.
    """
    championship_counts = defaultdict(int)

    for _ in range(n):
        champion = simulate_tournament(teams, elo_ratings)
        championship_counts[champion] += 1

    probs = {team: count / n for team, count in championship_counts.items()}
    return dict(sorted(probs.items(), key=lambda x: x[1], reverse=True))
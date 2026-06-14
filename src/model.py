import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, log_loss
import xgboost as xgb


def train_models(X_train, y_train):
    """
    Train and return all three calibrated models.
    """
    models = {
        'Logistic Regression': CalibratedClassifierCV(
            LogisticRegression(max_iter=1000)
        ),
        'Random Forest': CalibratedClassifierCV(
            RandomForestClassifier(n_estimators=100, random_state=42)
        ),
        'XGBoost': CalibratedClassifierCV(
            xgb.XGBClassifier(n_estimators=100, random_state=42,
                              eval_metric='mlogloss')
        )
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        print(f"Trained: {name}")

    return models


def evaluate_models(models, X_test, y_test):
    """
    Evaluate all models and return results dict sorted by log loss.
    """
    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        acc = accuracy_score(y_test, y_pred)
        ll = log_loss(y_test, y_proba)
        results[name] = {'accuracy': acc, 'log_loss': ll, 'model': model}
        print(f"{name}: Accuracy={acc:.3f}, Log Loss={ll:.3f}")

    best = min(results, key=lambda x: results[x]['log_loss'])
    print(f"\nBest model: {best}")

    return results, best


def save_model(model, path='data/processed/best_model.pkl'):
    """
    Save a trained model to disk.
    """
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {path}")


def load_model(path='data/processed/best_model.pkl'):
    """
    Load a trained model from disk.
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def predict_match(model, elo_ratings, team1, team2, neutral=1):
    """
    Predict win probabilities for a single match.
    Returns (team1_win, draw, team2_win) probabilities.
    """
    import pandas as pd

    elo1 = elo_ratings.get(team1, 1000)
    elo2 = elo_ratings.get(team2, 1000)

    features = pd.DataFrame([{
        'home_elo': elo1,
        'away_elo': elo2,
        'elo_diff': elo1 - elo2,
        'home_form': 0.5,
        'away_form': 0.5,
        'form_diff': 0.0,
        'neutral': neutral,
        'is_wc': 1
    }])

    probs = model.predict_proba(features)[0]
    # probs order: [away_win, draw, home_win]
    return probs[2], probs[1], probs[0]
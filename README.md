# xPredict | FIFA World Cup 2026 Predictor

A machine learning pipeline that predicts FIFA World Cup 2026 match outcomes and simulates the entire tournament bracket using 150+ years of international football data.

**Live app:** https://xpredict-wc2026.streamlit.app

---

## What it does

- **Championship Odds** - See which teams are most likely to win the 2026 World Cup based on 10,000 Monte Carlo simulations
- **Match Predictor** - Select any two teams and get win/draw/loss probabilities powered by a trained ML model
- **Tournament Simulator** - Simulate the entire 48-team bracket and explore different outcomes

---

## How it works

### Data
- 49,000+ international football results from 1872 to 2026 (Kaggle)
- FIFA World Cup 2026 fixture schedule and team data

### Feature Engineering
- **Elo ratings** - Built from scratch by processing every match chronologically since 1872
- **Recent form** - Win rate over each team's last 10 matches
- **Tournament context** - Whether the match is at a World Cup, on neutral ground, etc.

### Modeling
Three models were trained and compared on post-1990 match data:

| Model | Accuracy | Log Loss |
|---|---|---|
| Logistic Regression | 60.5% | 0.877 |
| XGBoost | 59.4% | 0.896 |
| Random Forest | 59.2% | 0.911 |

Logistic Regression was selected as the best model and calibrated using `CalibratedClassifierCV`.

### Simulation
The Tournament Simulator runs a configurable number of Monte Carlo simulations (1,000 to 20,000) across all 48 teams using Elo-based win probabilities to estimate each team's championship odds.

---

## Project structure
worldcup-2026-predictor/
├── data/
│   ├── raw/               # Downloaded datasets (not committed)
│   └── processed/         # Engineered features, model, Elo ratings
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── features.py
│   ├── model.py
│   └── simulate.py
├── app.py                 # Streamlit dashboard
├── requirements.txt
└── README.md
---

## Setup

```bash
git clone https://github.com/rafiaauthoi/worldcup-2026-predictor.git
cd worldcup-2026-predictor
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

---

## What's next (v2)

- Player-level features: squad depth, top scorer form, injury status
- Live match result updates during the tournament
- Improved bracket simulation following the actual 2026 group stage format

---

Built by [Rafia Authoi](https://github.com/rafiaauthoi)
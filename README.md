# March Madness 2026 Bracket Simulator

This project builds a lightweight NCAA men’s basketball model using team-level stats and a
Monte Carlo bracket simulator. It pulls public statistics from Sports Reference, trains a
classifier to predict head-to-head matchups, and then simulates the NCAA tournament to
identify the highest expected-score brackets for 2026.

## Data sources

- Team and advanced stats: `https://www.sports-reference.com/cbb/`
- Tournament results (historical): `https://www.sports-reference.com/cbb/postseason/`

> **Note**
> Sports Reference politely asks for low-volume, non-abusive scraping. The downloader uses a
> short delay between requests and caches data locally so you do not re-download on each run.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Download data and train the model (defaults: train seasons 2012-2024)
python scripts/train_model.py --start-season 2012 --end-season 2024

# Build a 2026 field from the latest available season stats and simulate brackets
python scripts/simulate_brackets.py --target-season 2026 --sims 2000
```

The simulation writes the two best brackets (by expected scoring) to:

- `output/brackets_2026.json`
- `output/brackets_2026.txt`

## Files

- `scripts/download_data.py` – downloads team stats and tournament games.
- `scripts/train_model.py` – trains the matchup model and persists it to `output/model.joblib`.
- `scripts/simulate_brackets.py` – builds a field, runs Monte Carlo simulation, exports best brackets.
- `src/march_madness/` – reusable modules for data prep, modeling, and simulation.

## Assumptions for 2026

The 2026 tournament field and seeds are not published yet. The simulator therefore creates a
field using the latest available season stats, ranks teams by a blended efficiency metric, and
assigns seeds using a standard 1–16 snake draft across four regions. Replace the generated
field by providing your own CSV using `--field-csv`.

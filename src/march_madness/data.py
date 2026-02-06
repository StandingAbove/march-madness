from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://www.sports-reference.com/cbb"


@dataclass
class DownloadConfig:
    data_dir: Path = Path("data")
    sleep_seconds: float = 1.0

    def ensure_dir(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)


def _polite_get(url: str) -> str:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def download_team_stats(season: int, config: DownloadConfig) -> pd.DataFrame:
    """Download team-level basic and advanced stats for a season."""
    config.ensure_dir()
    cached_path = config.data_dir / f"team_stats_{season}.csv"
    if cached_path.exists():
        return pd.read_csv(cached_path)

    url = f"{BASE_URL}/seasons/{season}-school-stats.html"
    html = _polite_get(url)
    tables = pd.read_html(html)

    basic = next(
        table for table in tables if "School" in table.columns and "W" in table.columns
    )
    advanced_candidates = [
        table
        for table in tables
        if "School" in table.columns and "ORtg" in table.columns
    ]
    if advanced_candidates:
        advanced = advanced_candidates[0]
        merged = basic.merge(advanced, on="School", how="left", suffixes=("", "_adv"))
    else:
        merged = basic

    merged = merged[merged["School"] != "School"]
    merged["Season"] = season

    merged.to_csv(cached_path, index=False)
    time.sleep(config.sleep_seconds)
    return merged


def download_tournament_games(season: int, config: DownloadConfig) -> pd.DataFrame:
    """Download NCAA tournament game results for a season."""
    config.ensure_dir()
    cached_path = config.data_dir / f"tourney_games_{season}.csv"
    if cached_path.exists():
        return pd.read_csv(cached_path)

    url = f"{BASE_URL}/postseason/{season}-ncaa.html"
    html = _polite_get(url)
    tables = pd.read_html(html)

    game_table = None
    for table in tables:
        if {"W", "L"}.issubset(set(table.columns)):
            game_table = table
            break

    if game_table is None:
        raise ValueError(f"No tournament games found for {season} at {url}")

    game_table = game_table.rename(columns={"W": "Winner", "L": "Loser"})
    game_table = game_table[["Winner", "Loser"]].dropna()
    game_table["Season"] = season

    game_table.to_csv(cached_path, index=False)
    time.sleep(config.sleep_seconds)
    return game_table


def build_training_dataset(start_season: int, end_season: int, config: DownloadConfig) -> pd.DataFrame:
    """Create matchup rows from historical tournament games and team stats."""
    rows = []
    for season in range(start_season, end_season + 1):
        stats = download_team_stats(season, config)
        games = download_tournament_games(season, config)

        stats = stats.set_index("School")
        numeric_cols = stats.select_dtypes(include="number").columns
        for _, game in games.iterrows():
            winner = game["Winner"]
            loser = game["Loser"]
            if winner not in stats.index or loser not in stats.index:
                continue

            winner_stats = stats.loc[winner, numeric_cols]
            loser_stats = stats.loc[loser, numeric_cols]
            features = (winner_stats - loser_stats).to_dict()
            features["Result"] = 1
            features["Season"] = season
            rows.append(features)

            reversed_features = (loser_stats - winner_stats).to_dict()
            reversed_features["Result"] = 0
            reversed_features["Season"] = season
            rows.append(reversed_features)

    dataset = pd.DataFrame(rows).dropna(axis=1, how="all")
    return dataset


def build_field_from_stats(stats: pd.DataFrame, field_size: int = 64) -> pd.DataFrame:
    """Create a tournament field by ranking teams on blended efficiency metrics."""
    stats = stats.copy()
    stats = stats[stats["School"] != "School"]
    for col in ("ORtg", "DRtg", "SRS", "SOS"):
        if col not in stats.columns:
            stats[col] = 0.0

    stats["EfficiencyScore"] = (
        stats["ORtg"].astype(float)
        - stats["DRtg"].astype(float)
        + stats["SRS"].astype(float)
        + 0.5 * stats["SOS"].astype(float)
    )
    field = stats.sort_values("EfficiencyScore", ascending=False).head(field_size)
    return field[["School", "EfficiencyScore"]].reset_index(drop=True)

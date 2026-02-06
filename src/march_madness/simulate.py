from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from march_madness.bracket import build_first_round

ROUND_SCORES = {
    "Round of 64": 1,
    "Round of 32": 2,
    "Sweet 16": 4,
    "Elite 8": 8,
    "Final Four": 16,
    "Championship": 32,
}


@dataclass
class BracketResult:
    rounds: dict[str, list[str]]
    log_probability: float


def _team_vector(team: str, stats: pd.DataFrame, feature_cols: Iterable[str]) -> pd.Series:
    row = stats.loc[team].copy()
    missing = set(feature_cols) - set(row.index)
    for col in missing:
        row[col] = 0.0
    return row[list(feature_cols)]


def _match_probability(team_a: str, team_b: str, stats: pd.DataFrame, model, feature_cols):
    vec_a = _team_vector(team_a, stats, feature_cols)
    vec_b = _team_vector(team_b, stats, feature_cols)
    features = (vec_a - vec_b).to_frame().T
    return model.predict_proba(features)[0, 1]


def simulate_bracket_once(matchups, stats, model, feature_cols, rng: np.random.Generator):
    rounds = {
        "Round of 64": [],
        "Round of 32": [],
        "Sweet 16": [],
        "Elite 8": [],
        "Final Four": [],
        "Championship": [],
    }
    log_probability = 0.0

    current_round = [(region, team_a.school, team_b.school) for region, team_a, team_b in matchups]
    rounds["Round of 64"] = [team for _, team, _ in current_round]

    round_names = [
        "Round of 64",
        "Round of 32",
        "Sweet 16",
        "Elite 8",
        "Final Four",
        "Championship",
    ]

    for idx, round_name in enumerate(round_names):
        winners = []
        for region, team_a, team_b in current_round:
            prob = _match_probability(team_a, team_b, stats, model, feature_cols)
            pick_a = rng.random() < prob
            winner = team_a if pick_a else team_b
            winners.append((region, winner))
            log_probability += math.log(prob if pick_a else (1 - prob) + 1e-9)

        rounds[round_name] = [winner for _, winner in winners]
        if len(winners) == 1:
            break

        if idx < 3:
            next_round = []
            by_region = {}
            for region, team in winners:
                by_region.setdefault(region, []).append(team)
            for region, teams in by_region.items():
                for i in range(0, len(teams), 2):
                    next_round.append((region, teams[i], teams[i + 1]))
        else:
            teams = [team for _, team in winners]
            if len(teams) == 2:
                next_round = [("Championship", teams[0], teams[1])]
            else:
                next_round = [
                    ("Final Four", teams[0], teams[1]),
                    ("Final Four", teams[2], teams[3]),
                ]
        current_round = next_round

    return BracketResult(rounds=rounds, log_probability=log_probability)


def simulate_brackets(matchups, stats, model, feature_cols, sims: int = 2000, seed: int = 13):
    rng = np.random.default_rng(seed)
    results = []
    for _ in range(sims):
        results.append(simulate_bracket_once(matchups, stats, model, feature_cols, rng))

    results = sorted(results, key=lambda r: r.log_probability, reverse=True)
    return results[:2]


def format_bracket(bracket: BracketResult) -> str:
    lines = []
    for round_name, winners in bracket.rounds.items():
        lines.append(f"{round_name}:")
        for winner in winners:
            lines.append(f"  - {winner}")
    lines.append(f"Log probability: {bracket.log_probability:.2f}")
    return "\n".join(lines)

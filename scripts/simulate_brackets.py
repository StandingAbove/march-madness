from __future__ import annotations

import argparse
from pathlib import Path

import json
import pandas as pd

from march_madness.bracket import assign_seeds, build_first_round
from march_madness.data import DownloadConfig, build_field_from_stats, download_team_stats
from march_madness.model import load_model
from march_madness.simulate import format_bracket, simulate_brackets


def load_field(field_csv: Path) -> pd.DataFrame:
    field = pd.read_csv(field_csv)
    if {"School", "EfficiencyScore"}.issubset(field.columns):
        return field
    if "School" in field.columns:
        field["EfficiencyScore"] = 0.0
        return field[["School", "EfficiencyScore"]]
    raise ValueError("Field CSV must contain a School column")


def main():
    parser = argparse.ArgumentParser(description="Simulate 2026 NCAA brackets.")
    parser.add_argument("--target-season", type=int, default=2026)
    parser.add_argument("--model", type=Path, default=Path("output/model.joblib"))
    parser.add_argument("--sims", type=int, default=2000)
    parser.add_argument("--field-csv", type=Path)
    parser.add_argument("--output-json", type=Path, default=Path("output/brackets_2026.json"))
    parser.add_argument("--output-text", type=Path, default=Path("output/brackets_2026.txt"))
    args = parser.parse_args()

    config = DownloadConfig()
    if args.field_csv:
        field = load_field(args.field_csv)
        stats = download_team_stats(args.target_season - 1, config)
    else:
        stats = download_team_stats(args.target_season - 1, config)
        field = build_field_from_stats(stats)

    model_artifacts = load_model(args.model)

    stats = stats.set_index("School")
    slots = assign_seeds(field)
    matchups = build_first_round(slots)

    best_brackets = simulate_brackets(
        matchups,
        stats,
        model_artifacts.model,
        model_artifacts.feature_columns,
        sims=args.sims,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as handle:
        payload = [bracket.rounds for bracket in best_brackets]
        json.dump(payload, handle, indent=2)

    with args.output_text.open("w", encoding="utf-8") as handle:
        for idx, bracket in enumerate(best_brackets, start=1):
            handle.write(f"Bracket {idx}\n")
            handle.write(format_bracket(bracket))
            handle.write("\n\n")

    print(f"Saved brackets to {args.output_text}")


if __name__ == "__main__":
    main()

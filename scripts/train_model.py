from __future__ import annotations

import argparse
from pathlib import Path

from march_madness.data import DownloadConfig, build_training_dataset
from march_madness.model import save_model, train_model


def main():
    parser = argparse.ArgumentParser(description="Train a tournament matchup model.")
    parser.add_argument("--start-season", type=int, default=2012)
    parser.add_argument("--end-season", type=int, default=2024)
    parser.add_argument("--output", type=Path, default=Path("output/model.joblib"))
    args = parser.parse_args()

    config = DownloadConfig()
    dataset = build_training_dataset(args.start_season, args.end_season, config)
    artifacts = train_model(dataset)
    save_model(artifacts, args.output)
    print(f"Saved model to {args.output}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse

from march_madness.data import DownloadConfig, download_team_stats, download_tournament_games


def main():
    parser = argparse.ArgumentParser(description="Download NCAA team stats and tournament games.")
    parser.add_argument("--start-season", type=int, default=2012)
    parser.add_argument("--end-season", type=int, default=2024)
    args = parser.parse_args()

    config = DownloadConfig()
    for season in range(args.start_season, args.end_season + 1):
        print(f"Downloading {season} team stats...")
        download_team_stats(season, config)
        print(f"Downloading {season} tournament games...")
        download_tournament_games(season, config)


if __name__ == "__main__":
    main()

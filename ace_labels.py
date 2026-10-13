"""
ace_labels.py — Pull MLB pitching data and label the ace of each team's staff.

Ace: among starters with >= MIN_GS starts for a team in a season, the pitcher
with the highest fWAR (tiebreak: IP). One ace per team per season.

Limitation: FanGraphs gives traded pitchers one combined line (Team == "- - -").
Those lines are excluded, so mid-season trade acquisitions can't be labeled.

Outputs:
    data/pitching_seasons.parquet   raw season-level pitching stats (all SP/RP)
    data/ace_labels.parquet         starters with `is_ace` boolean column
"""

from pathlib import Path
import pandas as pd
from pybaseball import pitching_stats, cache

cache.enable()

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

SEASONS = [2018, 2019, 2021, 2022, 2023, 2024]  # skip 2020 (60-game season)
MIN_GS = 10
N_TEAMS = 30
TRADED = "- - -"  # FanGraphs team label for multi-team seasons


def load_pitching_data(seasons, refresh: bool = False) -> pd.DataFrame:
    """Pull FanGraphs season-level pitching stats and cache to parquet."""
    cache_path = DATA_DIR / "pitching_seasons.parquet"
    if cache_path.exists() and not refresh:
        print(f"Loading cached pitching data from {cache_path}")
        return pd.read_parquet(cache_path)

    frames = []
    for season in seasons:
        print(f"Fetching pitching_stats({season})...")
        df = pitching_stats(season, qual=0)  # all pitchers; GS filter applied later
        df["Season"] = season
        frames.append(df)

    all_data = pd.concat(frames, ignore_index=True)
    all_data.to_parquet(cache_path)
    print(f"Saved {len(all_data)} player-seasons -> {cache_path}")
    return all_data


def label_aces(df: pd.DataFrame, min_gs: int = MIN_GS) -> pd.DataFrame:
    """Mark the highest-WAR starter (tiebreak: IP) in each Team-Season."""
    # Combined multi-team lines aren't a real staff; keeping them creates a fake 31st team
    starters = df[(df["GS"] >= min_gs) & (df["Team"] != TRADED)].copy()

    starters = starters.sort_values(
        ["Season", "Team", "WAR", "IP"],
        ascending=[True, True, False, False],
    )
    starters["is_ace"] = starters.groupby(["Season", "Team"]).cumcount() == 0
    return starters


def sanity_check(raw: pd.DataFrame, labeled: pd.DataFrame, min_gs: int = MIN_GS) -> None:
    aces = labeled[labeled["is_ace"]]
    print("\n=== Sanity check ===")
    print(f"Total ace-seasons labeled: {len(aces)}")

    # Exactly one ace per team-season
    per_team = aces.groupby(["Season", "Team"]).size()
    assert (per_team == 1).all(), "Some team-season has more than one ace"

    # Flags both too few (team had no qualifying SP) and too many (bad team labels)
    teams_per_season = aces.groupby("Season")["Team"].nunique()
    off = teams_per_season[teams_per_season != N_TEAMS]
    if len(off):
        print(f"\nSeasons without exactly {N_TEAMS} ace-teams:")
        print(off)

    # Size of the traded-pitcher exclusion
    traded = raw[(raw["Team"] == TRADED) & (raw["GS"] >= min_gs)]
    print(f"\nTraded starters excluded (GS >= {min_gs}):")
    print(traded.groupby("Season").size())

    print("\nTop 2024 aces by WAR:")
    cols = ["Season", "Team", "Name", "GS", "IP", "WAR", "ERA", "FIP"]
    sample = (
        aces[aces["Season"] == 2024]
        .sort_values("WAR", ascending=False)[cols]
        .head(10)
    )
    print(sample.to_string(index=False))


if __name__ == "__main__":
    pitching = load_pitching_data(SEASONS)
    labeled = label_aces(pitching, min_gs=MIN_GS)
    labeled.to_parquet(DATA_DIR / "ace_labels.parquet")
    sanity_check(pitching, labeled)

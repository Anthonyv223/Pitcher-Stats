"""
ace_labels.py — Pull MLB pitching data and label the ace of each team's staff.

Defines an "ace" as: among starters with >= MIN_GS starts for a given team
in a season, the pitcher with the highest fWAR. Tiebreaker: more IP.
One ace per team per season.

Outputs:
    data/pitching_seasons.parquet   raw season-level pitching stats (all SP/RP)
    data/ace_labels.parquet         starters with `is_ace` boolean column
"""

from pathlib import Path
import pandas as pd
from pybaseball import pitching_stats, cache

# Use pybaseball's local cache so re-runs don't re-download
cache.enable()

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

SEASONS = [2018, 2019, 2021, 2022, 2023, 2024]  # skip 2020 (60-game season)
MIN_GS = 10


def load_pitching_data(seasons, refresh: bool = False) -> pd.DataFrame:
    """Pull FanGraphs season-level pitching stats and cache to parquet."""
    cache_path = DATA_DIR / "pitching_seasons.parquet"
    if cache_path.exists() and not refresh:
        print(f"Loading cached pitching data from {cache_path}")
        return pd.read_parquet(cache_path)

    frames = []
    for season in seasons:
        print(f"Fetching pitching_stats({season})...")
        # qual=0 returns every pitcher; we filter by GS ourselves
        df = pitching_stats(season, qual=0)
        df["Season"] = season
        frames.append(df)

    all_data = pd.concat(frames, ignore_index=True)
    all_data.to_parquet(cache_path)
    print(f"Saved {len(all_data)} player-seasons -> {cache_path}")
    return all_data


def label_aces(df: pd.DataFrame, min_gs: int = MIN_GS) -> pd.DataFrame:
    """
    For each Team-Season, mark the ace as the SP with highest WAR (tiebreak: IP).
    Only pitchers with GS >= min_gs are considered.
    """
    starters = df[df["GS"] >= min_gs].copy()

    # Sort so the ace lands first within each Team-Season group
    starters = starters.sort_values(
        ["Season", "Team", "WAR", "IP"],
        ascending=[True, True, False, False],
    )

    starters["is_ace"] = (
        starters.groupby(["Season", "Team"]).cumcount() == 0
    )
    return starters


def sanity_check(labeled: pd.DataFrame) -> None:
    aces = labeled[labeled["is_ace"]]
    print("\n=== Sanity check ===")
    print(f"Total ace-seasons labeled: {len(aces)}")
    print("\nAces labeled per season:")
    print(aces.groupby("Season").size())

    # Flag seasons missing an ace for any team
    teams_per_season = aces.groupby("Season")["Team"].nunique()
    missing = teams_per_season[teams_per_season < 30]
    if len(missing):
        print("\nSeasons without 30 distinct ace-teams "
              "(some team had no SP with >= MIN_GS):")
        print(missing)

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
    sanity_check(labeled)
statcast_features.py
python"""
statcast_features.py — Stage 2.

Reads data/ace_labels.parquet (from ace_labels.py), pulls Statcast pitch-level
data for every labeled starter-season, engineers stuff/command features, and
merges them with the FanGraphs season stats into one analysis table.

Outputs:
    data/statcast_raw/<mlbam>_<season>.parquet   per-pitcher-season raw pitches (cache)
    data/statcast_features.parquet               engineered features per pitcher-season
    data/analysis_table.parquet                  FanGraphs stats + Statcast features merged

NOTE: Statcast pulls are slow (one player-season per request). The per-file
cache makes this resumable — kill and restart any time; finished files skip.
"""

from pathlib import Path
import time
import numpy as np
import pandas as pd
from pybaseball import statcast_pitcher, playerid_reverse_lookup, cache

cache.enable()

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "statcast_raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Regular-season date ranges (Opening Day -> last day of regular season)
SEASON_DATES = {
    2018: ("2018-03-29", "2018-10-01"),
    2019: ("2019-03-20", "2019-09-29"),
    2021: ("2021-04-01", "2021-10-03"),
    2022: ("2022-04-07", "2022-10-05"),
    2023: ("2023-03-30", "2023-10-01"),
    2024: ("2024-03-28", "2024-09-30"),
}

# Pitch-result description buckets (Statcast `description` field)
WHIFF = {"swinging_strike", "swinging_strike_blocked"}
CSW = {"called_strike", "swinging_strike", "swinging_strike_blocked"}
SWING = WHIFF | {"foul", "foul_tip", "hit_into_play", "foul_bunt"}
FASTBALLS = {"FF", "SI", "FC"}  # four-seam, sinker, cutter


def get_mlbam_ids(labeled: pd.DataFrame) -> pd.DataFrame:
    """Map FanGraphs IDs (IDfg) to MLBAM IDs needed by statcast_pitcher."""
    fg_ids = labeled["IDfg"].dropna().astype(int).unique().tolist()
    lookup = playerid_reverse_lookup(fg_ids, key_type="fangraphs")
    out = lookup[["key_fangraphs", "key_mlbam"]].dropna()
    out["key_fangraphs"] = out["key_fangraphs"].astype(int)
    out["key_mlbam"] = out["key_mlbam"].astype(int)
    return out


def pull_raw_statcast(merged: pd.DataFrame) -> None:
    """Download (and cache) raw pitch data for each pitcher-season."""
    total = len(merged)
    for i, row in enumerate(merged.itertuples(), 1):
        mlbam, season = int(row.key_mlbam), int(row.Season)
        out_path = RAW_DIR / f"{mlbam}_{season}.parquet"
        if out_path.exists():
            continue
        start, end = SEASON_DATES[season]
        try:
            df = statcast_pitcher(start, end, mlbam)
            df.to_parquet(out_path)
            print(f"[{i}/{total}] {row.Name} {season}: {len(df)} pitches")
        except Exception as e:
            print(f"[{i}/{total}] FAILED {row.Name} {season} ({mlbam}): {e}")
        time.sleep(1)  # be polite to the server


def compute_features(p: pd.DataFrame) -> dict:
    """Engineer stuff/command features from one pitcher-season of raw pitches."""
    n = len(p)
    if n == 0:
        return {}

    desc = p["description"]
    swings = desc.isin(SWING).sum()
    whiffs = desc.isin(WHIFF).sum()
    csw = desc.isin(CSW).sum()

    # Plate discipline (zone 1-9 = in zone; 11-14 = out of zone)
    zoned = p["zone"].notna()
    in_zone = p["zone"].between(1, 9)
    out_zone = zoned & ~in_zone
    swing_mask = desc.isin(SWING)
    o_pitches = out_zone.sum()
    o_swings = (swing_mask & out_zone).sum()

    # Fastball velocity / shape
    fb = p[p["pitch_type"].isin(FASTBALLS)]

    # Repertoire
    mix = p["pitch_type"].value_counts(normalize=True)
    n_types = int((mix >= 0.05).sum())          # pitches used >=5% of the time
    entropy = float(-(mix * np.log2(mix)).sum()) if len(mix) else 0.0

    # Contact quality (over batted balls)
    bip = p["launch_speed"].notna()
    ev = p.loc[bip, "launch_speed"]

    return {
        "pitches": n,
        "csw_pct": csw / n,
        "whiff_pct": whiffs / swings if swings else np.nan,
        "chase_pct": o_swings / o_pitches if o_pitches else np.nan,
        "zone_pct": in_zone.sum() / zoned.sum() if zoned.sum() else np.nan,
        "fb_velo": fb["release_speed"].mean(),
        "fb_velo_max": fb["release_speed"].max(),
        "fb_spin": fb["release_spin_rate"].mean(),
        "avg_spin": p["release_spin_rate"].mean(),
        "avg_extension": p["release_extension"].mean(),
        "n_pitch_types": n_types,
        "pitch_entropy": entropy,
        "avg_exit_velo": ev.mean() if len(ev) else np.nan,
        "hard_hit_pct": (ev >= 95).mean() if len(ev) else np.nan,
    }


def build_feature_table(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in merged.itertuples():
        mlbam, season = int(row.key_mlbam), int(row.Season)
        path = RAW_DIR / f"{mlbam}_{season}.parquet"
        if not path.exists():
            continue
        feats = compute_features(pd.read_parquet(path))
        if not feats:
            continue
        feats.update({
            "Season": season,
            "key_mlbam": mlbam,
            "IDfg": int(row.IDfg),
            "Name": row.Name,
            "Team": row.Team,
            "is_ace": bool(row.is_ace),
        })
        rows.append(feats)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    labeled = pd.read_parquet(DATA_DIR / "ace_labels.parquet")

    id_map = get_mlbam_ids(labeled)
    merged = labeled.merge(id_map, left_on="IDfg", right_on="key_fangraphs")
    print(f"Resolved MLBAM IDs for {len(merged)} of {len(labeled)} starter-seasons")

    pull_raw_statcast(merged)

    features = build_feature_table(merged)
    features.to_parquet(DATA_DIR / "statcast_features.parquet")
    print(f"\nBuilt Statcast features for {len(features)} pitcher-seasons")

    # Merge Statcast features with the FanGraphs season stats for modeling
    fg_cols = ["IDfg", "Season", "ERA", "FIP", "xFIP", "WHIP", "K/9", "BB/9",
               "K-BB%", "IP", "GS", "WAR"]
    fg_cols = [c for c in fg_cols if c in labeled.columns]
    analysis = features.merge(labeled[fg_cols], on=["IDfg", "Season"], how="left")
    analysis.to_parquet(DATA_DIR / "analysis_table.parquet")
    print(f"Saved analysis_table.parquet ({analysis.shape[0]} rows, "
          f"{analysis.shape[1]} cols)")

    print("\nAce vs non-ace feature means:")
    show = ["fb_velo", "whiff_pct", "csw_pct", "chase_pct",
            "avg_extension", "n_pitch_types", "hard_hit_pct"]
    show = [c for c in show if c in analysis.columns]
    print(analysis.groupby("is_ace")[show].mean().round(3).T)

"""
statcast_features.py — Stage 2.
 
Reads data/ace_labels.parquet, pulls Statcast pitch-level data for each labeled
starter-season, builds stuff/command features from regular-season starts only,
and compares each ace to his own rotation-mates.
 
Outputs:
    data/statcast_raw_v2/<mlbam>_<season>.parquet   raw pitches per pitcher-season (cache)
    data/statcast_features.parquet                  engineered features per pitcher-season
    data/analysis_table.parquet                     features + rotation-relative features + FG context
 
Resumable: finished player-season files are skipped on re-run.
"""
 
from pathlib import Path
import time
import numpy as np
import pandas as pd
from pybaseball import statcast_pitcher, playerid_reverse_lookup, cache
 
cache.enable()
 
DATA_DIR = Path("data")
# New cache dir: v1 used hardcoded Opening Days and missed the 2024 Seoul Series
RAW_DIR = DATA_DIR / "statcast_raw_v2"
RAW_DIR.mkdir(parents=True, exist_ok=True)
 
# Statcast `description` buckets
WHIFF = {"swinging_strike", "swinging_strike_blocked", "foul_tip", "missed_bunt"}  # Savant counts foul tips as whiffs
CSW = WHIFF | {"called_strike"}
IN_PLAY = {"hit_into_play", "hit_into_play_no_out", "hit_into_play_score"}  # last two: pre-2019 labels
SWING = WHIFF | IN_PLAY | {"foul", "foul_bunt", "bunt_foul_tip"}
 
PRIMARY_FB = ["FF", "SI"]  # cutters excluded: slower by design, would drag velo down
BREAKING = ["SL", "ST", "SV", "CU", "KC"]
 
# Explanatory features (Statcast only)
FEATURES = [
    "csw_pct", "whiff_pct", "chase_pct", "zone_pct",
    "fb_velo", "fb_velo_max", "fb_spin", "bb_spin", "avg_extension",
    "n_pitch_types", "pitch_entropy", "avg_exit_velo", "hard_hit_pct",
]
 
# fWAR is built from FIP + IP, so these overlap the label. Context only, never predictors.
FG_CONTEXT = ["ERA", "FIP", "xFIP", "WHIP", "K/9", "BB/9", "K-BB%", "IP", "GS", "WAR"]
 
 
def season_window(season: int) -> tuple[str, str]:
    # Wide window; regular season is isolated later via game_type
    return f"{season}-03-01", f"{season}-11-30"
 
 
def get_mlbam_ids(labeled: pd.DataFrame) -> pd.DataFrame:
    """Map FanGraphs IDs to MLBAM IDs; report any that don't resolve."""
    fg_ids = labeled["IDfg"].unique().tolist()
    lookup = playerid_reverse_lookup(fg_ids, key_type="fangraphs")
    out = lookup[["key_fangraphs", "key_mlbam"]].dropna().astype(int)
    out = out.drop_duplicates("key_fangraphs")  # duplicates would double rows on merge
 
    missing = set(fg_ids) - set(out["key_fangraphs"])
    if missing:
        print(f"Unresolved MLBAM IDs ({len(missing)} players):")
        cols = ["Name", "Season", "Team", "is_ace"]
        print(labeled.loc[labeled["IDfg"].isin(missing), cols].to_string(index=False))
    return out
 
 
def pull_raw_statcast(merged: pd.DataFrame) -> None:
    """Download and cache raw pitch data for each pitcher-season."""
    total = len(merged)
    for i, row in enumerate(merged.itertuples(), 1):
        mlbam, season = int(row.key_mlbam), int(row.Season)
        out_path = RAW_DIR / f"{mlbam}_{season}.parquet"
        if out_path.exists():
            continue
        start, end = season_window(season)
        try:
            df = statcast_pitcher(start, end, mlbam)
            df.to_parquet(out_path)
            print(f"[{i}/{total}] {row.Name} {season}: {len(df)} pitches")
        except Exception as e:
            print(f"[{i}/{total}] FAILED {row.Name} {season} ({mlbam}): {e}")
        time.sleep(1)
 
 
def regular_season_starts(p: pd.DataFrame) -> pd.DataFrame:
    """Keep regular-season games the pitcher started."""
    if "game_type" in p.columns:
        p = p[p["game_type"] == "R"]  # drops spring training and postseason
    # Start = pitched in the 1st inning. Removes relief outings, which inflate velo.
    # Bulk innings behind an opener are also dropped.
    started = p.loc[p["inning"] == 1, "game_pk"].unique()
    return p[p["game_pk"].isin(started)]
 
 
def primary_pitch(p: pd.DataFrame, types: list[str]):
    """Most-thrown pitch type among `types`, or None."""
    counts = p.loc[p["pitch_type"].isin(types), "pitch_type"].value_counts()
    return counts.idxmax() if len(counts) else None
 
 
def compute_features(p: pd.DataFrame) -> dict:
    """Stuff/command features from one pitcher-season of starts."""
    n = len(p)
    if n == 0:
        return {}
 
    desc = p["description"]
    swing_mask = desc.isin(SWING)
    swings = swing_mask.sum()
    whiffs = desc.isin(WHIFF).sum()
    csw = desc.isin(CSW).sum()
 
    # Zones 1-9 in, 11-14 out
    zoned = p["zone"].notna()
    in_zone = p["zone"].between(1, 9)
    out_zone = zoned & ~in_zone
    o_pitches = out_zone.sum()
    o_swings = (swing_mask & out_zone).sum()
 
    # Primary fastball only; fall back to cutter for cutter-first pitchers
    fb_type = primary_pitch(p, PRIMARY_FB) or primary_pitch(p, ["FC"])
    fb = p[p["pitch_type"] == fb_type] if fb_type else p.iloc[0:0]
 
    # Spin per pitch type; an all-pitch average mostly measures mix
    bb_type = primary_pitch(p, BREAKING)
    bb = p[p["pitch_type"] == bb_type] if bb_type else p.iloc[0:0]
 
    mix = p["pitch_type"].dropna().value_counts(normalize=True)
    n_types = int((mix >= 0.05).sum())  # pitches used >= 5% of the time
    entropy = float(-(mix * np.log2(mix)).sum()) if len(mix) else 0.0
 
    # Balls in play only; launch_speed can be populated on fouls
    ev = p.loc[p["type"] == "X", "launch_speed"].dropna()
 
    return {
        "pitches": n,
        "starts": p["game_pk"].nunique(),
        "csw_pct": csw / n,
        "whiff_pct": whiffs / swings if swings else np.nan,
        "chase_pct": o_swings / o_pitches if o_pitches else np.nan,
        "zone_pct": in_zone.sum() / zoned.sum() if zoned.sum() else np.nan,
        "fb_type": fb_type,
        "fb_velo": fb["release_speed"].mean(),
        "fb_velo_max": fb["release_speed"].max(),
        "fb_spin": fb["release_spin_rate"].mean(),
        "bb_type": bb_type,
        "bb_spin": bb["release_spin_rate"].mean(),
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
        feats = compute_features(regular_season_starts(pd.read_parquet(path)))
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
 
 
def add_rotation_relative(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """
    <col>_rel = pitcher's value minus his rotation-mates' mean (self excluded).
    Matches the question: #1 vs #2-5 on the same staff, not vs the league.
    """
    df = df.copy()
    g = df.groupby(["Season", "Team"])
    for c in cols:
        total = g[c].transform("sum")
        count = g[c].transform("count")
        mates_mean = (total - df[c]) / (count - 1)  # NaN when no mates have data
        df[f"{c}_rel"] = df[c] - mates_mean
    return df
 
 
if __name__ == "__main__":
    labeled = pd.read_parquet(DATA_DIR / "ace_labels.parquet")
    labeled = labeled.dropna(subset=["IDfg"])
    labeled["IDfg"] = labeled["IDfg"].astype(int)
 
    id_map = get_mlbam_ids(labeled)
    merged = labeled.merge(id_map, left_on="IDfg", right_on="key_fangraphs")
    print(f"Resolved MLBAM IDs for {len(merged)} of {len(labeled)} starter-seasons")
 
    pull_raw_statcast(merged)
 
    features = build_feature_table(merged)
    features.to_parquet(DATA_DIR / "statcast_features.parquet")
    print(f"\nBuilt Statcast features for {len(features)} pitcher-seasons")
 
    # Paired comparison needs the ace present; drop team-seasons where his data is missing
    has_ace = features.groupby(["Season", "Team"])["is_ace"].transform("any")
    dropped = features.loc[~has_ace, ["Season", "Team"]].drop_duplicates()
    if len(dropped):
        print(f"Dropped {len(dropped)} team-seasons with no ace Statcast data")
    features = features[has_ace]
 
    analysis = add_rotation_relative(features, FEATURES)
 
    fg_cols = [c for c in FG_CONTEXT if c in labeled.columns]
    analysis = analysis.merge(labeled[["IDfg", "Season"] + fg_cols],
                              on=["IDfg", "Season"], how="left")
    analysis.to_parquet(DATA_DIR / "analysis_table.parquet")
    print(f"Saved analysis_table.parquet ({analysis.shape[0]} rows, "
          f"{analysis.shape[1]} cols)")
 
    # Ace minus his own rotation-mates, averaged across team-seasons
    rel = [f"{c}_rel" for c in FEATURES]
    aces = analysis[analysis["is_ace"]]
    summary = pd.DataFrame({
        "ace_minus_mates": aces[rel].mean(),
        "pct_aces_higher": aces[rel].apply(lambda s: (s.dropna() > 0).mean()),
        "n": aces[rel].notna().sum(),
    })
    summary.index = summary.index.str.replace("_rel", "", regex=False)
    print("\nAce vs. own rotation-mates:")
    print(summary.round(3))
 

"""
Re-applies the current scoring logic to all existing driver/constructor pickles
without hitting the FastF1 API. Use this after changing scoring rules.

Run: python rescoreCache.py
     python rescoreCache.py --dry-run   (preview only, no writes)
"""
import os
import sys
import pandas as pd

from raceCalendars import (
    races_2019, races_2020, races_2021, races_2022,
    races_2023, races_2024, races_2025, races_2026
)
from Scoring.preprocess import preprocess_driver_data
from Scoring.driverFantasyPoints import calculate_fantasy_points
from Scoring.constructorFantasyPoints import calculate_constructor_points

CACHE_DIR = "cache/pickles"

SEASON_RACE_MAP = {
    2019: races_2019,
    2020: races_2020,
    2021: races_2021,
    2022: races_2022,
    2023: races_2023,
    2024: races_2024,
    2025: races_2025,
    2026: races_2026,
}

# Columns produced by scoring — drop these and recompute
SCORING_COLS = [
    'DNF', 'positions_gained', 'sprint_positions_gained', 'sprint_dnf',
    'sprint_disqualified', 'qualifying_disqualified',
    'qualifying_points',
    'sprint_points', 'sprint_gain_points', 'sprint_fastest_lap_bonus',
    'sprint_overtakes', 'sprint_overtake_points', 'sprint_dnf_penalty',
    'sprint_dsq_penalty', 'sprint_total',
    'race_points', 'race_gain_points', 'race_fastest_lap_bonus',
    'dotd_bonus', 'race_overtake_points', 'race_dnf_penalty',
    'fantasy_points_total',
]


def rescore_round(season, round_no, dry_run=False):
    driver_path = f"{CACHE_DIR}/driver_{season}_r{round_no}.pkl"
    constructor_path = f"{CACHE_DIR}/constructor_{season}_r{round_no}.pkl"

    if not os.path.exists(driver_path):
        return False, "driver pickle not found"

    try:
        driver_df = pd.read_pickle(driver_path)
    except Exception as e:
        return False, f"failed to load: {e}"

    # Resolve overtakes_x / overtakes_y collision from old pipeline (telemetry merged after scoring)
    if 'overtakes_x' in driver_df.columns and 'overtakes_y' in driver_df.columns:
        # overtakes_y is from telemetry (real data); overtakes_x was the zeroed placeholder
        driver_df['overtakes'] = driver_df['overtakes_y'].fillna(0)
        driver_df = driver_df.drop(columns=['overtakes_x', 'overtakes_y'])
    elif 'overtakes_x' in driver_df.columns:
        driver_df = driver_df.rename(columns={'overtakes_x': 'overtakes'})
    elif 'overtakes_y' in driver_df.columns:
        driver_df = driver_df.rename(columns={'overtakes_y': 'overtakes'})

    # Drop stale scoring columns (keep raw input columns)
    cols_to_drop = [c for c in SCORING_COLS if c in driver_df.columns]
    raw_df = driver_df.drop(columns=cols_to_drop)

    # Re-run preprocessing and scoring
    try:
        clean_df = preprocess_driver_data(raw_df)
        scored_df = calculate_fantasy_points(clean_df)
    except Exception as e:
        return False, f"scoring failed: {e}"

    # Re-derive constructor points from pitstop time stored in existing constructor pickle
    pitstop_df = pd.DataFrame(columns=['constructor_name', 'fastest_pitstop_time'])
    if os.path.exists(constructor_path):
        try:
            old_constructor_df = pd.read_pickle(constructor_path)
            if 'fastest_pitstop_time' in old_constructor_df.columns:
                pitstop_df = old_constructor_df[['constructor_name', 'fastest_pitstop_time']].dropna()
        except Exception:
            pass

    try:
        new_constructor_df = calculate_constructor_points(scored_df, pitstop_df)
        # Restore metadata columns from old constructor pickle
        if os.path.exists(constructor_path):
            old_con = pd.read_pickle(constructor_path)
            for meta_col in ['season', 'round', 'event_name', 'had_sprint',
                             'track_temp_avg', 'air_temp_avg', 'humidity_avg',
                             'pressure_avg', 'rainfall_avg', 'wind_speed_avg', 'wind_direction_avg']:
                if meta_col in old_con.columns:
                    # Same value for all rows — take first
                    val = old_con[meta_col].iloc[0]
                    new_constructor_df[meta_col] = val
    except Exception as e:
        return False, f"constructor scoring failed: {e}"

    if not dry_run:
        scored_df.to_pickle(driver_path)
        new_constructor_df.to_pickle(constructor_path)

    old_total = driver_df['fantasy_points_total'].sum() if 'fantasy_points_total' in driver_df.columns else None
    new_total = scored_df['fantasy_points_total'].sum()
    delta = (new_total - old_total) if old_total is not None else None
    return True, f"points delta={delta:+.1f}" if delta is not None else "rescored"


def main(dry_run=False):
    if dry_run:
        print("DRY RUN — no files will be written.\n")

    rescored = 0
    skipped = 0
    errors = 0

    for season, rounds in SEASON_RACE_MAP.items():
        season_results = []
        for r in rounds:
            ok, msg = rescore_round(season, r, dry_run=dry_run)
            if ok:
                rescored += 1
                season_results.append(f"r{r}: {msg}")
            elif "not found" in msg:
                skipped += 1
            else:
                errors += 1
                season_results.append(f"r{r} ERROR: {msg}")

        if season_results:
            print(f"{season}: {', '.join(season_results)}")

    print(f"\nDone. Rescored={rescored}  Skipped(not cached)={skipped}  Errors={errors}")
    if dry_run:
        print("Re-run without --dry-run to apply changes.")


if __name__ == '__main__':
    dry_run = '--dry-run' in sys.argv
    main(dry_run=dry_run)

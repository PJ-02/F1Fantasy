"""
Backfills telemetry data (overtakes, top_speed, avg_throttle, brake_usage)
into existing driver pickles using the FastF1 disk cache — no API calls needed.
Also re-runs scoring so race_overtake_points are correctly calculated.

Run: python backfillTelemetry.py
     python backfillTelemetry.py --dry-run
"""
import os
import sys
import fastf1
import pandas as pd

from raceCalendars import (
    races_2019, races_2020, races_2021, races_2022,
    races_2023, races_2024, races_2025, races_2026
)
from DataRetrieval.telemetry import get_driver_telemetry
from Scoring.preprocess import preprocess_driver_data
from Scoring.driverFantasyPoints import calculate_fantasy_points
from Utils.common import logger, get_session_with_retry, load_session_with_retry

fastf1.Cache.enable_cache('cache/f1_cache')

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


def needs_telemetry_backfill(driver_df):
    """True if top_speed is entirely NaN (telemetry was never merged)."""
    return 'top_speed' not in driver_df.columns or driver_df['top_speed'].isna().all()


def backfill_round(season, round_no, dry_run=False):
    driver_path = f"{CACHE_DIR}/driver_{season}_r{round_no}.pkl"
    if not os.path.exists(driver_path):
        return 'skip', "not cached"

    driver_df = pd.read_pickle(driver_path)

    if not needs_telemetry_backfill(driver_df):
        return 'skip', "telemetry already present"

    # Load session from FastF1 disk cache
    try:
        session = get_session_with_retry(season, round_no, 'R')
        session = load_session_with_retry(session)
    except Exception as e:
        return 'error', f"session load failed: {e}"

    telemetry_df = get_driver_telemetry(session)
    if telemetry_df.empty or 'driver_id' not in telemetry_df.columns:
        return 'error', "telemetry unavailable in FastF1 cache"

    # Resolve any leftover overtakes_x/y from old pipeline
    if 'overtakes_x' in driver_df.columns and 'overtakes_y' in driver_df.columns:
        driver_df = driver_df.drop(columns=['overtakes_x', 'overtakes_y'])
    elif 'overtakes_x' in driver_df.columns:
        driver_df = driver_df.rename(columns={'overtakes_x': 'overtakes'})
    elif 'overtakes_y' in driver_df.columns:
        driver_df = driver_df.rename(columns={'overtakes_y': 'overtakes'})

    # Drop stale telemetry columns if any exist
    telemetry_cols = ['top_speed', 'avg_throttle', 'brake_usage', 'overtakes']
    driver_df = driver_df.drop(columns=[c for c in telemetry_cols if c in driver_df.columns])

    # Merge fresh telemetry
    driver_df = driver_df.merge(telemetry_df, on='driver_id', how='left')

    # Drop stale scoring columns and re-score with correct overtake data
    cols_to_drop = [c for c in SCORING_COLS if c in driver_df.columns]
    raw_df = driver_df.drop(columns=cols_to_drop)

    try:
        clean_df = preprocess_driver_data(raw_df)
        scored_df = calculate_fantasy_points(clean_df)
    except Exception as e:
        return 'error', f"re-scoring failed: {e}"

    drivers_with_telemetry = telemetry_df['driver_id'].nunique()
    avg_overtakes = telemetry_df['overtakes'].mean()

    if not dry_run:
        scored_df.to_pickle(driver_path)

    return 'ok', f"{drivers_with_telemetry} drivers, avg_overtakes={avg_overtakes:.1f}"


def main(dry_run=False):
    if dry_run:
        print("DRY RUN — no files will be written.\n")

    filled = skipped = errors = 0

    for season, rounds in SEASON_RACE_MAP.items():
        season_results = []
        for r in rounds:
            status, msg = backfill_round(season, r, dry_run=dry_run)
            if status == 'ok':
                filled += 1
                season_results.append(f"r{r}: {msg}")
            elif status == 'error':
                errors += 1
                season_results.append(f"r{r} ERROR: {msg}")
            else:
                skipped += 1

        if season_results:
            print(f"{season}: {', '.join(season_results)}")

    print(f"\nDone. Filled={filled}  Skipped={skipped}  Errors={errors}")
    if dry_run:
        print("Re-run without --dry-run to apply changes.")


if __name__ == '__main__':
    dry_run = '--dry-run' in sys.argv
    main(dry_run=dry_run)

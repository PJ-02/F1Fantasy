"""
Backfills accurate stationary pit stop times into existing constructor pickles
using the OpenF1 API.

OpenF1's stop_duration field is available from the 2024 United States Grand Prix
(Austin, Round 19) onwards. For earlier rounds the script skips silently.

Re-running is safe — if OpenF1 returns no data for a round it is skipped.

Usage:
    python backfillPitstops.py                        # all seasons
    python backfillPitstops.py --seasons 2024 2025 2026
    python backfillPitstops.py --seasons 2024 --dry-run
"""

import os
import sys
import argparse
import pandas as pd

from raceCalendars import (
    races_2019, races_2020, races_2021, races_2022,
    races_2023, races_2024, races_2025, races_2026,
)
from DataRetrieval.pitstops_openf1 import get_openf1_pitstops
from Scoring.constructorFantasyPoints import calculate_constructor_points
from Utils.common import logger

CACHE_DIR = 'cache/pickles'

SEASON_RACE_MAP = {
    2019: races_2019, 2020: races_2020, 2021: races_2021, 2022: races_2022,
    2023: races_2023, 2024: races_2024, 2025: races_2025, 2026: races_2026,
}


def backfill_round(season: int, round_no: int, dry_run: bool = False):
    driver_path      = f"{CACHE_DIR}/driver_{season}_r{round_no}.pkl"
    constructor_path = f"{CACHE_DIR}/constructor_{season}_r{round_no}.pkl"

    if not os.path.exists(constructor_path):
        return 'skip', "constructor pickle not cached"
    if not os.path.exists(driver_path):
        return 'skip', "driver pickle not cached"

    openf1_df = get_openf1_pitstops(season, round_no)
    if openf1_df.empty:
        return 'skip', "no OpenF1 stop_duration data (pre-Austin 2024 or unavailable)"

    # Re-score constructor points with correct pitstop times
    driver_df = pd.read_pickle(driver_path)
    constructor_df = pd.read_pickle(constructor_path)

    # Recalculate constructor points with accurate pit times
    new_constructor_df = calculate_constructor_points(driver_df, openf1_df)

    # Preserve metadata columns from original pickle that aren't in scoring output
    meta_cols = ['season', 'round', 'event_name', 'had_sprint',
                 'track_temp_avg', 'air_temp_avg', 'humidity_avg',
                 'pressure_avg', 'rainfall_avg', 'wind_speed_avg', 'wind_direction_avg']
    for col in meta_cols:
        if col in constructor_df.columns and col not in new_constructor_df.columns:
            # Merge from original on constructor_name
            mapping = constructor_df.set_index('constructor_name')[col]
            new_constructor_df[col] = new_constructor_df['constructor_name'].map(mapping)
            # For season/round/event_name/had_sprint use the first value (same for all rows)
            if col in ['season', 'round', 'event_name', 'had_sprint']:
                new_constructor_df[col] = constructor_df[col].iloc[0]

    old_total = constructor_df['constructor_fantasy_points'].sum()
    new_total = new_constructor_df['constructor_fantasy_points'].sum()

    if not dry_run:
        new_constructor_df.to_pickle(constructor_path)

    delta = new_total - old_total
    fastest = openf1_df['fastest_pitstop_time'].min()
    return 'ok', (
        f"fastest stop {fastest:.3f}s  |  "
        f"total pts {old_total:.0f} → {new_total:.0f} (Δ{delta:+.0f})"
    )


def main(seasons=None, dry_run=False):
    if dry_run:
        print("DRY RUN — no files will be written.\n")

    target = {s: r for s, r in SEASON_RACE_MAP.items()
              if seasons is None or s in seasons}

    ok_count = skipped = errors = 0

    for season, rounds in sorted(target.items()):
        season_msgs = []
        for r in rounds:
            try:
                status, msg = backfill_round(season, r, dry_run=dry_run)
            except Exception as e:
                status, msg = 'error', str(e)

            if status == 'ok':
                ok_count += 1
                season_msgs.append(f"  r{r}: {msg}")
            elif status == 'error':
                errors += 1
                season_msgs.append(f"  r{r} ERROR: {msg}")
            else:
                skipped += 1

        if season_msgs:
            print(f"\n{season}:")
            print('\n'.join(season_msgs))
        else:
            print(f"{season}: no rounds updated")

    print(f"\nDone.  Updated={ok_count}  Skipped={skipped}  Errors={errors}")
    if dry_run:
        print("Re-run without --dry-run to apply changes.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Backfill OpenF1 stationary pit times into constructor pickles'
    )
    parser.add_argument('--seasons', type=int, nargs='+',
                        help='Seasons to process (default: all). E.g. --seasons 2024 2025 2026')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview changes without writing')
    args = parser.parse_args()
    main(seasons=set(args.seasons) if args.seasons else None, dry_run=args.dry_run)

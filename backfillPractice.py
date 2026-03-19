"""
Backfills practice session pace data (fp1_gap_pct, fp2_gap_pct, fp3_gap_pct,
fp1_laps, fp2_laps, fp3_laps) into existing driver pickles.

Only processes rounds that are missing practice columns — safe to re-run.
Reads from the FastF1 disk cache, so no redundant API calls for already-
cached sessions. Sessions not in the cache will trigger fresh API calls
(subject to rate limits).

Usage:
    python backfillPractice.py                   # all seasons
    python backfillPractice.py --seasons 2024 2025 2026
    python backfillPractice.py --dry-run
"""

import os
import sys
import argparse
import fastf1
import pandas as pd

from raceCalendars import (
    races_2019, races_2020, races_2021, races_2022,
    races_2023, races_2024, races_2025, races_2026,
)
from DataRetrieval.practice import get_practice_pace
from Utils.common import logger

fastf1.Cache.enable_cache('cache/f1_cache')

CACHE_DIR = 'cache/pickles'

SEASON_RACE_MAP = {
    2019: races_2019, 2020: races_2020, 2021: races_2021, 2022: races_2022,
    2023: races_2023, 2024: races_2024, 2025: races_2025, 2026: races_2026,
}

PRACTICE_COLS = ['fp1_gap_pct', 'fp2_gap_pct', 'fp3_gap_pct',
                 'fp1_laps',    'fp2_laps',    'fp3_laps']


def needs_backfill(driver_df: pd.DataFrame) -> bool:
    """True if no practice columns are present yet."""
    return not any(c in driver_df.columns for c in PRACTICE_COLS)


def backfill_round(season: int, round_no: int, dry_run: bool = False):
    driver_path = f"{CACHE_DIR}/driver_{season}_r{round_no}.pkl"
    if not os.path.exists(driver_path):
        return 'skip', "not cached"

    driver_df = pd.read_pickle(driver_path)
    if not needs_backfill(driver_df):
        return 'skip', "practice data already present"

    # Drop any stale partial practice columns from previous failed attempts
    driver_df = driver_df.drop(
        columns=[c for c in PRACTICE_COLS if c in driver_df.columns]
    )

    filled = []
    for fp_type in ['FP1', 'FP2', 'FP3']:
        fp_df = get_practice_pace(season, round_no, fp_type)
        prefix = fp_type.lower()
        if not fp_df.empty and 'driver_id' in fp_df.columns:
            driver_df = driver_df.merge(fp_df, on='driver_id', how='left')
            filled.append(fp_type)
        else:
            # Session doesn't exist (sprint weekend) or unavailable — store NaN
            driver_df[f'{prefix}_gap_pct'] = float('nan')
            driver_df[f'{prefix}_laps']    = float('nan')

    if not dry_run:
        driver_df.to_pickle(driver_path)

    sessions_str = ','.join(filled) if filled else 'none available'
    return 'ok', f"filled {sessions_str}"


def main(seasons=None, dry_run=False):
    if dry_run:
        print("DRY RUN — no files will be written.\n")

    target = {s: r for s, r in SEASON_RACE_MAP.items()
              if seasons is None or s in seasons}

    ok_count = skipped = errors = 0

    for season, rounds in sorted(target.items()):
        season_msgs = []
        for r in rounds:
            status, msg = backfill_round(season, r, dry_run=dry_run)
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
            print(f"{season}: all rounds already up to date")

    print(f"\nDone.  Filled={ok_count}  Skipped={skipped}  Errors={errors}")
    if dry_run:
        print("Re-run without --dry-run to apply changes.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backfill practice pace into driver pickles')
    parser.add_argument('--seasons', type=int, nargs='+',
                        help='Seasons to process (default: all). E.g. --seasons 2024 2025 2026')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview without writing')
    args = parser.parse_args()
    main(seasons=set(args.seasons) if args.seasons else None, dry_run=args.dry_run)

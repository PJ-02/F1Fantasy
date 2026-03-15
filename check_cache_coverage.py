"""
Audits the pickle cache to show which rounds are cached, which are missing,
and flags any quality issues (NaN columns, low driver counts) in cached data.
"""
import os
import sys
import pandas as pd

from raceCalendars import (
    races_2019, races_2020, races_2021, races_2022,
    races_2023, races_2024, races_2025, races_2026
)

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

# Columns expected in every driver pickle
REQUIRED_DRIVER_COLS = [
    'driver_id', 'driver_name', 'constructor_name', 'grid', 'position',
    'status', 'fastest_lap', 'qualifying_pos',
    'DNF', 'positions_gained', 'qualifying_disqualified',
    'qualifying_points', 'race_points', 'fantasy_points_total',
    'season', 'round', 'event_name',
]

# Sprint columns — only expected to be non-NaN when had_sprint=True
SPRINT_COLS = ['sprint_pos', 'sprint_grid', 'sprint_status']

# Telemetry/weather columns that are optional (NaN is acceptable)
OPTIONAL_COLS = [
    'top_speed', 'avg_throttle', 'brake_usage', 'overtakes',
    'track_temp_avg', 'air_temp_avg', 'humidity_avg',
    'pressure_avg', 'rainfall_avg', 'wind_speed_avg', 'wind_direction_avg',
]


def audit_driver_pickle(path):
    """Returns a dict of issues found in a driver pickle."""
    issues = []
    try:
        df = pd.read_pickle(path)
    except Exception as e:
        return {'load_error': str(e)}

    # Check driver count
    n = len(df)
    if n < 18:
        issues.append(f"only {n} drivers (expected 20)")

    # Check required columns
    missing_cols = [c for c in REQUIRED_DRIVER_COLS if c not in df.columns]
    if missing_cols:
        issues.append(f"missing columns: {missing_cols}")

    # Check NaN rates for required columns
    high_nan = []
    for col in REQUIRED_DRIVER_COLS:
        if col in df.columns:
            nan_pct = df[col].isna().mean()
            if nan_pct > 0.5:
                high_nan.append(f"{col}={nan_pct:.0%}NaN")
    if high_nan:
        issues.append(f"high NaN in required cols: {high_nan}")

    # Check sprint columns only for sprint rounds
    had_sprint = df['had_sprint'].any() if 'had_sprint' in df.columns else False
    if had_sprint:
        sprint_nan = []
        for col in SPRINT_COLS:
            if col in df.columns:
                nan_pct = df[col].isna().mean()
                if nan_pct > 0.5:
                    sprint_nan.append(f"{col}={nan_pct:.0%}NaN")
        if sprint_nan:
            issues.append(f"sprint round but high NaN in sprint cols: {sprint_nan}")

    # Report NaN rates for optional columns
    optional_nan = []
    for col in OPTIONAL_COLS:
        if col in df.columns:
            nan_pct = df[col].isna().mean()
            if nan_pct > 0:
                optional_nan.append(f"{col}={nan_pct:.0%}")

    return {
        'rows': n,
        'issues': issues,
        'optional_nan': optional_nan,
        'columns': list(df.columns),
    }


def main(verbose=False):
    total_missing = []
    total_issues = []

    for season, rounds in SEASON_RACE_MAP.items():
        cached = []
        missing = []
        issues_in_season = []

        for r in rounds:
            driver_path = f"{CACHE_DIR}/driver_{season}_r{r}.pkl"
            constructor_path = f"{CACHE_DIR}/constructor_{season}_r{r}.pkl"

            driver_exists = os.path.exists(driver_path)
            constructor_exists = os.path.exists(constructor_path)

            if driver_exists and constructor_exists:
                cached.append(r)
                audit = audit_driver_pickle(driver_path)
                if audit.get('issues'):
                    issues_in_season.append((r, audit['issues']))
                    total_issues.append((season, r, audit['issues']))
                if verbose and audit.get('optional_nan'):
                    issues_in_season.append((r, [f"optional NaN: {audit['optional_nan']}"]))
            else:
                missing.append(r)
                total_missing.append((season, r))
                if driver_exists and not constructor_exists:
                    missing[-1] = f"r{r}(no constructor pkl)"
                elif not driver_exists and constructor_exists:
                    missing[-1] = f"r{r}(no driver pkl)"

        status = "✓ complete" if not missing else f"✗ missing {len(missing)}/{len(rounds)}"
        print(f"\n{season}  [{status}]  cached={len(cached)}")
        if missing:
            print(f"  MISSING rounds: {missing}")
        if issues_in_season:
            for r, msgs in issues_in_season:
                print(f"  Round {r} ISSUES: {'; '.join(msgs)}")

    print("\n" + "="*60)
    print(f"SUMMARY")
    print(f"  Total missing rounds : {len(total_missing)}")
    if total_missing:
        by_season = {}
        for s, r in total_missing:
            by_season.setdefault(s, []).append(r)
        for s, rs in sorted(by_season.items()):
            print(f"    {s}: rounds {rs}")
    print(f"  Rounds with issues   : {len(total_issues)}")
    for s, r, msgs in total_issues:
        print(f"    {s} r{r}: {'; '.join(msgs)}")

    if total_missing:
        print("\nRun `python mainPipeline.py` to collect missing rounds.")
    if not total_missing and not total_issues:
        print("\nAll cached data looks clean.")


if __name__ == '__main__':
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    main(verbose=verbose)

"""
Predict fantasy points for an upcoming F1 race — two modes:

  pre  (--mode pre)   Before qualifying: estimates qualifying positions from
                      rolling history, then predicts total fantasy points.

  post (--mode post)  After qualifying:  uses actual qualifying positions to
                      compute exact qualifying pts, then predicts race pts.
                      Total = known Q pts + predicted race pts.

Usage:
  python predictRace.py --season 2026 --round 3 --mode pre
  python predictRace.py --season 2026 --round 3 --mode post
  python predictRace.py --season 2026 --round 3              # defaults to post
  python predictRace.py --season 2026 --round 3 --no-save

Output:
  Ranked predictions printed to stdout.
  Saved to predictions/{season}_r{round}_{mode}_predictions.csv
"""

import os
import sys
import argparse
import warnings

import joblib
import numpy as np
import pandas as pd
import fastf1

warnings.filterwarnings('ignore')

from trainModel import engineer_features, FEATURE_COLS
from Utils.common import get_session_with_retry, load_session_with_retry

CACHE_DIR  = 'cache/pickles'
MODELS_DIR = 'models'
PREDS_DIR  = 'predictions'

SEASON_LENGTHS = {
    2019: 21, 2020: 14, 2021: 22, 2022: 22,
    2023: 22, 2024: 24, 2025: 24, 2026: 24,
}

# Qualifying position → fantasy points (deterministic rule)
QUALI_PTS_MAP = {1: 10, 2: 9, 3: 8, 4: 7, 5: 6, 6: 5, 7: 4, 8: 3, 9: 2, 10: 1}


def quali_pts_from_pos(pos) -> float:
    """Return qualifying fantasy points for a given finishing position."""
    if pd.isna(pos):
        return -5.0  # no time / DNS treated as penalty
    return float(QUALI_PTS_MAP.get(int(pos), 0))


# ---------------------------------------------------------------------------
# Constructor helpers
# ---------------------------------------------------------------------------

def compute_qualifying_bonus(positions: list) -> int:
    """
    Compute the constructor qualifying bonus from a list of driver qualifying
    positions (typically two drivers per constructor).

    Bonus rules (mirrors constructorFantasyPoints.py):
      q2_count = drivers with pos <= 15  →  0→-1, 1→+1, 2→+3
      q3_count = drivers with pos <= 10  →  1→+5, 2→+10  (additive)
    """
    q2_count = sum(1 for p in positions if pd.notna(p) and int(p) <= 15)
    q3_count = sum(1 for p in positions if pd.notna(p) and int(p) <= 10)

    bonus = -1 if q2_count == 0 else (1 if q2_count == 1 else 3)
    bonus += (5 if q3_count == 1 else 10 if q3_count >= 2 else 0)
    return bonus


def build_constructor_output(driver_output: pd.DataFrame,
                              driver_pts_col: str,
                              mode: str) -> pd.DataFrame:
    """
    Aggregate per-driver predictions to constructor level.

    driver_output must contain: constructor_name, driver_name,
                                qualifying_pos, <driver_pts_col>

    Returns a sorted DataFrame with one row per constructor.
    """
    rows = []
    for constructor, group in driver_output.groupby('constructor_name'):
        positions   = group['qualifying_pos'].tolist()
        q_bonus     = compute_qualifying_bonus(positions)
        driver_pts  = group[driver_pts_col].sum()
        total       = driver_pts + q_bonus
        driver_list = ' / '.join(group['driver_name'].tolist())

        rows.append({
            'constructor_name':  constructor,
            'drivers':           driver_list,
            'qualifying_bonus':  q_bonus,
            'pred_driver_pts':   round(float(driver_pts), 1),
            'pred_total':        round(float(total), 1),
        })

    out = (pd.DataFrame(rows)
           .sort_values('pred_total', ascending=False)
           .reset_index(drop=True))
    out.index += 1
    return out


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def load_history() -> pd.DataFrame:
    dfs = []
    for fname in sorted(os.listdir(CACHE_DIR)):
        if fname.startswith('driver_') and fname.endswith('.pkl'):
            dfs.append(pd.read_pickle(os.path.join(CACHE_DIR, fname)))
    return pd.concat(dfs, ignore_index=True)


def get_weather(session) -> dict:
    try:
        w = session.weather_data
        return {
            'track_temp_avg': float(w['TrackTemp'].mean()),
            'air_temp_avg':   float(w['AirTemp'].mean()),
            'humidity_avg':   float(w['Humidity'].mean()),
            'pressure_avg':   float(w['Pressure'].mean()),
            'rainfall_avg':   float(w['Rainfall'].mean()),
            'wind_speed_avg': float(w['WindSpeed'].mean()),
        }
    except Exception:
        return {k: np.nan for k in
                ['track_temp_avg', 'air_temp_avg', 'humidity_avg',
                 'pressure_avg',   'rainfall_avg', 'wind_speed_avg']}


def check_sprint(season: int, round_no: int) -> bool:
    try:
        event = fastf1.get_event(season, round_no)
        for col in event.index:
            if 'Session' in str(col) and 'sprint' in str(event[col]).lower():
                return True
        return False
    except Exception:
        return False


def _build_and_engineer(history: pd.DataFrame, upcoming_rows: list,
                         season: int, round_no: int) -> pd.DataFrame:
    """
    Append synthetic upcoming-race rows to history, engineer features,
    then return only the target round rows.
    The target round is stripped from history first to avoid duplicates
    when predicting on an already-cached round (e.g. backtesting).
    """
    history = history[
        ~((history['season'] == season) & (history['round'] == round_no))
    ].copy()

    upcoming = pd.DataFrame(upcoming_rows)
    combined  = pd.concat([history, upcoming], ignore_index=True)
    engineered = engineer_features(combined)

    mask = (engineered['season'] == season) & (engineered['round'] == round_no)
    pred_rows = engineered[mask].copy()

    # Fix round_pct — engineer_features uses max(round in df) which underestimates
    # the season length when the season is still in progress
    season_len = SEASON_LENGTHS.get(season, 24)
    pred_rows['round_pct'] = round_no / season_len

    return pred_rows


def _predict_and_format(model, pred_rows: pd.DataFrame) -> pd.Series:
    """Run model, return a Series of predicted values indexed by driver_id."""
    X = pred_rows[FEATURE_COLS]
    preds = model.predict(X)
    return pd.Series(preds, index=pred_rows['driver_id'].values)


# ---------------------------------------------------------------------------
# Mode 1: Pre-qualifying
# ---------------------------------------------------------------------------

def predict_pre_qualifying(model, history: pd.DataFrame,
                            season: int, round_no: int,
                            event_name: str, had_sprint: bool) -> pd.DataFrame:
    """
    Estimate qualifying positions from each driver's rolling 3-race quali
    average. Drivers with no history fall back to the median estimated pos.
    Estimated qualifying pts are computed directly from the predicted rank.
    Model predicts total fantasy pts (using estimated positions as input).
    """
    # --- Driver lineup: take from the most recent cached round this season,
    #     falling back to the previous season if none exist yet.
    season_pickles = sorted([
        f for f in os.listdir(CACHE_DIR)
        if f.startswith(f'driver_{season}_') and f.endswith('.pkl')
    ])
    if not season_pickles:
        prev_season_pickles = sorted([
            f for f in os.listdir(CACHE_DIR)
            if f.startswith(f'driver_{season - 1}_') and f.endswith('.pkl')
        ])
        latest_pkl = prev_season_pickles[-1] if prev_season_pickles else None
    else:
        latest_pkl = season_pickles[-1]

    if latest_pkl is None:
        print("ERROR: No cached rounds found to derive driver lineup.")
        sys.exit(1)

    latest_df = pd.read_pickle(os.path.join(CACHE_DIR, latest_pkl))
    lineup_cols = ['driver_id', 'driver_name', 'constructor_name']
    lineup = latest_df[lineup_cols].drop_duplicates('driver_id').copy()
    print(f"  Driver lineup sourced from: {latest_pkl}")

    # --- Estimate qualifying positions from rolling 3-race quali average ---
    # Compute driver_quali_rolling3 from history (same logic as engineer_features)
    hist_sorted = history.sort_values(['driver_id', 'season', 'round'])
    grp = hist_sorted.groupby('driver_id')
    rolling_quali = (
        grp['qualifying_pos']
        .apply(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    hist_sorted = hist_sorted.copy()
    hist_sorted['_rolling_quali'] = rolling_quali

    # Take the most recent value per driver
    latest_quali = (
        hist_sorted.sort_values(['season', 'round'])
        .groupby('driver_id')['_rolling_quali']
        .last()
    )
    lineup['est_quali_pos_raw'] = lineup['driver_id'].map(latest_quali)

    # Fill unknown drivers (rookies / no history) with median
    median_pos = lineup['est_quali_pos_raw'].median()
    if np.isnan(median_pos):
        median_pos = 11.0
    lineup['est_quali_pos_raw'] = lineup['est_quali_pos_raw'].fillna(median_pos)

    # Rank to get integer positions 1–N (no ties)
    lineup['qualifying_pos'] = (
        lineup['est_quali_pos_raw']
        .rank(method='first')
        .astype(int)
    )

    # Estimated qualifying fantasy pts from predicted rank
    lineup['est_quali_pts'] = lineup['qualifying_pos'].map(
        lambda p: QUALI_PTS_MAP.get(p, 0)
    ).astype(float)

    # --- Build synthetic rows and engineer features ---
    rows = []
    for _, driver in lineup.iterrows():
        row = {
            'driver_id':            driver['driver_id'],
            'driver_name':          driver['driver_name'],
            'constructor_name':     driver['constructor_name'],
            'season':               season,
            'round':                round_no,
            'event_name':           event_name,
            'qualifying_pos':       float(driver['qualifying_pos']),
            'grid':                 float(driver['qualifying_pos']),
            'had_sprint':           had_sprint,
            'fantasy_points_total': np.nan,
            'DNF':                  0,
            'position':             np.nan,
            'status':               '',
            'sprint_pos':           np.nan,
            # No weather — let imputer handle NaN
            **{k: np.nan for k in ['track_temp_avg', 'air_temp_avg', 'humidity_avg',
                                    'pressure_avg', 'rainfall_avg', 'wind_speed_avg']},
        }
        rows.append(row)

    pred_rows = _build_and_engineer(history, rows, season, round_no)
    pred_rows['predicted_total'] = _predict_and_format(model, pred_rows).values

    # Merge estimated quali pts back in
    pred_rows = pred_rows.merge(
        lineup[['driver_id', 'qualifying_pos', 'est_quali_pts']],
        on='driver_id', how='left', suffixes=('', '_est')
    )
    # Use the estimated qualifying_pos (integer rank) for display
    pred_rows['qualifying_pos'] = pred_rows['qualifying_pos_est'].fillna(
        pred_rows['qualifying_pos']
    )

    output = pred_rows[[
        'driver_id', 'driver_name', 'constructor_name',
        'qualifying_pos', 'est_quali_pts', 'predicted_total',
    ]].sort_values('predicted_total', ascending=False).reset_index(drop=True)
    output.index += 1
    output['qualifying_pos']  = output['qualifying_pos'].astype(int)
    output['est_quali_pts']   = output['est_quali_pts'].round(1)
    output['predicted_total'] = output['predicted_total'].round(1)

    return output


# ---------------------------------------------------------------------------
# Mode 2: Post-qualifying
# ---------------------------------------------------------------------------

def fetch_qualifying(season: int, round_no: int):
    """
    Returns (DataFrame, session).
    DataFrame columns: driver_id, driver_name, qualifying_pos, constructor_name
    """
    session = get_session_with_retry(season, round_no, 'Q')
    session = load_session_with_retry(session, telemetry=False, laps=False)

    res = session.results[['Abbreviation', 'FullName', 'Position', 'TeamName']].copy()
    res.columns = ['driver_id', 'driver_name', 'qualifying_pos', 'constructor_name']
    res['driver_id']      = res['driver_id'].str.lower()
    res['qualifying_pos'] = res['qualifying_pos'].astype(float)
    return res.reset_index(drop=True), session


def predict_post_qualifying(model, history: pd.DataFrame,
                             season: int, round_no: int,
                             event_name: str, had_sprint: bool,
                             quali_df: pd.DataFrame,
                             weather: dict) -> pd.DataFrame:
    """
    Actual qualifying positions are known.
    Qualifying fantasy pts are computed exactly from those positions.
    Model predicts total; race pts = model total - actual quali pts.
    Final total = actual quali pts + predicted race pts.
    """
    # Compute exact qualifying fantasy pts
    quali_df = quali_df.copy()
    quali_df['actual_quali_pts'] = quali_df['qualifying_pos'].map(quali_pts_from_pos)

    # Build synthetic rows
    rows = []
    for _, driver in quali_df.iterrows():
        row = {
            'driver_id':            driver['driver_id'],
            'driver_name':          driver['driver_name'],
            'constructor_name':     driver['constructor_name'],
            'season':               season,
            'round':                round_no,
            'event_name':           event_name,
            'qualifying_pos':       float(driver['qualifying_pos']),
            'grid':                 float(driver['qualifying_pos']),  # assume no grid penalties
            'had_sprint':           had_sprint,
            'fantasy_points_total': np.nan,
            'DNF':                  0,
            'position':             np.nan,
            'status':               '',
            'sprint_pos':           np.nan,
        }
        row.update(weather)
        rows.append(row)

    pred_rows = _build_and_engineer(history, rows, season, round_no)
    pred_rows['model_total'] = _predict_and_format(model, pred_rows).values

    # Merge actual quali pts in
    pred_rows = pred_rows.merge(
        quali_df[['driver_id', 'qualifying_pos', 'actual_quali_pts']],
        on='driver_id', how='left', suffixes=('', '_actual')
    )
    pred_rows['qualifying_pos'] = pred_rows['qualifying_pos_actual'].fillna(
        pred_rows['qualifying_pos']
    )

    # Race prediction = model total (which includes Q pts in its reasoning) minus Q pts
    pred_rows['pred_race_pts']  = pred_rows['model_total'] - pred_rows['actual_quali_pts']
    pred_rows['pred_total']     = pred_rows['actual_quali_pts'] + pred_rows['pred_race_pts']

    output = pred_rows[[
        'driver_id', 'driver_name', 'constructor_name',
        'qualifying_pos', 'actual_quali_pts', 'pred_race_pts', 'pred_total',
    ]].sort_values('pred_total', ascending=False).reset_index(drop=True)
    output.index += 1
    output['qualifying_pos']  = output['qualifying_pos'].astype(int)
    output['actual_quali_pts'] = output['actual_quali_pts'].round(1)
    output['pred_race_pts']   = output['pred_race_pts'].round(1)
    output['pred_total']      = output['pred_total'].round(1)

    return output


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Predict F1 fantasy points before or after qualifying'
    )
    parser.add_argument('--season',  type=int,   required=True)
    parser.add_argument('--round',   type=int,   required=True)
    parser.add_argument('--mode',    type=str,   default='post',
                        choices=['pre', 'post'],
                        help='"pre" = before qualifying  |  "post" = after qualifying (default)')
    parser.add_argument('--no-save', action='store_true')
    args = parser.parse_args()

    season, round_no, mode = args.season, args.round, args.mode

    fastf1.Cache.enable_cache('cache/f1_cache')

    # --- Load model ---
    model_path = os.path.join(MODELS_DIR, 'fantasy_model.joblib')
    if not os.path.exists(model_path):
        print("ERROR: No trained model found. Run python trainModel.py first.")
        sys.exit(1)
    model = joblib.load(model_path)
    print(f"Model loaded: {model_path}")
    print(f"Mode: {'PRE-qualifying (estimating Q positions)' if mode == 'pre' else 'POST-qualifying (using actual Q results)'}\n")

    # --- Event metadata (no session load needed) ---
    event      = fastf1.get_event(season, round_no)
    event_name = event['EventName']
    had_sprint = check_sprint(season, round_no)
    print(f"Event: {event_name} {season} R{round_no}  |  Sprint weekend: {had_sprint}")

    # --- History ---
    print("Loading historical data...")
    history = load_history()
    print(f"  {len(history)} rows across {history['season'].nunique()} seasons\n")

    # --- Run the selected mode ---
    if mode == 'pre':
        driver_output = predict_pre_qualifying(
            model, history, season, round_no, event_name, had_sprint
        )
        driver_pts_col   = 'predicted_total'
        driver_col_rename = {
            'driver_id':       'ID',
            'driver_name':     'Driver',
            'constructor_name':'Constructor',
            'qualifying_pos':  'Est.Q.Pos',
            'est_quali_pts':   'Est.Q.Pts',
            'predicted_total': 'Est.Total',
        }
        con_note    = "(Q.Bonus from estimated positions; Pred.Driver.Pts = sum of driver Est.Totals)"
        driver_note = "(Est.Q.Pts from estimated rank; Est.Total from model)"

    else:  # post
        print(f"Fetching qualifying results for {season} R{round_no}...")
        try:
            quali_df, quali_session = fetch_qualifying(season, round_no)
        except Exception as e:
            print(f"ERROR: Could not fetch qualifying data — {e}")
            sys.exit(1)

        if quali_df.empty:
            print(f"ERROR: No qualifying data found for {season} R{round_no}.")
            print("Qualifying may not have taken place yet — try --mode pre instead.")
            sys.exit(1)

        print(f"  {len(quali_df)} drivers")
        weather = get_weather(quali_session)
        if not np.isnan(weather.get('air_temp_avg', np.nan)):
            print(f"  Weather: {weather['air_temp_avg']:.1f}°C air, "
                  f"{weather['track_temp_avg']:.1f}°C track, "
                  f"rain={weather['rainfall_avg']:.1f}%\n")
        else:
            print("  Weather: unavailable — model will use median imputation\n")

        driver_output = predict_post_qualifying(
            model, history, season, round_no, event_name, had_sprint,
            quali_df, weather
        )
        driver_pts_col   = 'pred_total'
        driver_col_rename = {
            'driver_id':        'ID',
            'driver_name':      'Driver',
            'constructor_name': 'Constructor',
            'qualifying_pos':   'Q.Pos',
            'actual_quali_pts': 'Q.Pts (actual)',
            'pred_race_pts':    'Pred.Race.Pts',
            'pred_total':       'Pred.Total',
        }
        con_note    = "(Q.Bonus exact from actual positions; Pred.Driver.Pts = sum of both driver Pred.Totals)"
        driver_note = "(Q.Pts exact from actual positions; Pred.Race.Pts from model)"

    # --- Constructor predictions ---
    con_output = build_constructor_output(driver_output, driver_pts_col, mode)
    con_col_rename = {
        'constructor_name': 'Constructor',
        'drivers':          'Drivers',
        'qualifying_bonus': 'Q.Bonus',
        'pred_driver_pts':  'Pred.Driver.Pts',
        'pred_total':       'Pred.Total',
    }

    label = 'PRE' if mode == 'pre' else 'POST'

    # --- Print drivers ---
    header = f"  {event_name} {season} R{round_no} — {label}-QUALIFYING · DRIVERS  "
    print(f"\n{'=' * len(header)}")
    print(header)
    print(f"{'=' * len(header)}")
    print(driver_output.rename(columns=driver_col_rename).to_string())
    print(f"\n{driver_note}")

    # --- Print constructors ---
    con_header = f"  {event_name} {season} R{round_no} — {label}-QUALIFYING · CONSTRUCTORS  "
    print(f"\n{'=' * len(con_header)}")
    print(con_header)
    print(f"{'=' * len(con_header)}")
    print(con_output.rename(columns=con_col_rename).to_string())
    print(f"\n{con_note}")
    print("  Note: Pitstop bonuses not predicted (known data limitation — always stored as 0)")

    # --- Save ---
    if not args.no_save:
        os.makedirs(PREDS_DIR, exist_ok=True)

        driver_path = os.path.join(PREDS_DIR, f'{season}_r{round_no}_{mode}_driver_predictions.csv')
        driver_output.to_csv(driver_path)
        print(f"\nSaved: {driver_path}")

        con_path = os.path.join(PREDS_DIR, f'{season}_r{round_no}_{mode}_constructor_predictions.csv')
        con_output.to_csv(con_path)
        print(f"Saved: {con_path}")


if __name__ == '__main__':
    main()

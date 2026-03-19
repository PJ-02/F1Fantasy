"""
Trains a regression model to predict driver fantasy_points_total for an upcoming race.
Only features available BEFORE the race starts are used (qualifying results + rolling history).

Outputs:
  models/fantasy_model.joblib       — trained pipeline (best model)
  models/feature_columns.json       — ordered list of feature names the model expects
  models/model_eval.txt             — evaluation report

Usage:
  python trainModel.py
  python trainModel.py --eval-only  (print evaluation without saving)
"""

import os
import sys
import json
import joblib
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings('ignore')

CACHE_DIR   = 'cache/pickles'
MODELS_DIR  = 'models'
TRAIN_SEASONS = [2019, 2020, 2021, 2022, 2023, 2024]
TEST_SEASONS  = [2025]
# 2026 held out entirely — used only for live prediction

BASELINE_MAE = 8.83   # RandomForest v2 baseline (tested on 2024)

# ---------------------------------------------------------------------------
# STATIC LOOKUP TABLES
# ---------------------------------------------------------------------------

# Driver birth years — used to compute driver_age at race time.
# Verified for all 38 drivers in the 2019–2026 dataset.
DRIVER_BIRTH_YEAR = {
    'ham': 1985, 'alo': 1981, 'rai': 1979, 'kub': 1984,
    'gro': 1986, 'vet': 1987, 'hul': 1987, 'bot': 1989,
    'ric': 1989, 'per': 1990, 'mag': 1992, 'gio': 1993,
    'sai': 1994, 'kvy': 1994, 'dev': 1995, 'lat': 1995,
    'gas': 1996, 'oco': 1996, 'alb': 1996, 'ver': 1997,
    'lec': 1997, 'rus': 1998, 'str': 1998, 'maz': 1999,
    'msc': 1999, 'zho': 1999, 'nor': 1999, 'tsu': 2000,
    'sar': 2000, 'pia': 2001, 'law': 2002, 'col': 2003,
    'doo': 2003, 'bor': 2004, 'had': 2004, 'bea': 2005,
    'lin': 2005, 'ant': 2006,
}

# Street circuits: tight, wall-lined tracks where car/driver errors are punished
# harder and overtaking is more difficult.
STREET_CIRCUITS = {
    'Azerbaijan Grand Prix',
    'Las Vegas Grand Prix',
    'Miami Grand Prix',
    'Monaco Grand Prix',
    'Saudi Arabian Grand Prix',
    'Singapore Grand Prix',
}

# High-altitude circuits: thinner air significantly changes car behaviour
# (reduced downforce, engine cooling). Teams adapt differently.
HIGH_ALTITUDE_CIRCUITS = {
    'Mexican Grand Prix',
    'Mexico City Grand Prix',
}


# ---------------------------------------------------------------------------
# 1. LOAD ALL PICKLES
# ---------------------------------------------------------------------------

def load_all_data():
    dfs = []
    for fname in sorted(os.listdir(CACHE_DIR)):
        if fname.startswith('driver_') and fname.endswith('.pkl'):
            dfs.append(pd.read_pickle(os.path.join(CACHE_DIR, fname)))
    df = pd.concat(dfs, ignore_index=True)
    df = df.sort_values(['season', 'round']).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
#    All rolling features use .shift(1) — strictly no-leakage.
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # --- Normalise constructor names across rebrands ---
    rebrand_map = {
        'Alfa Romeo Racing': 'Alfa Romeo',
        'AlphaTauri':        'RB',
        'Racing Bulls':      'RB',
        'Toro Rosso':        'RB',
        'Renault':           'Alpine',
        'Racing Point':      'Aston Martin',
        'Kick Sauber':       'Alfa Romeo',
    }
    df['constructor_norm'] = df['constructor_name'].replace(rebrand_map)

    # Numeric DNF flag
    df['DNF'] = df['DNF'].astype(int)

    # Race index for stable chronological ordering
    df['race_idx'] = df['season'] * 100 + df['round']

    # =========================================================
    # DRIVER ROLLING FEATURES
    # =========================================================
    df = df.sort_values(['driver_id', 'race_idx']).reset_index(drop=True)
    grp = df.groupby('driver_id')

    # --- Points: multiple timescales ---
    df['driver_pts_rolling3'] = (
        grp['fantasy_points_total']
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    df['driver_pts_rolling5'] = (
        grp['fantasy_points_total']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    df['driver_pts_rolling10'] = (
        grp['fantasy_points_total']
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    )
    # Exponential weighted mean — recent races count more than older ones
    df['driver_pts_ewm5'] = (
        grp['fantasy_points_total']
        .transform(lambda x: x.shift(1).ewm(span=5, min_periods=1).mean())
    )

    # --- Form momentum: short-term vs long-term avg (positive = improving) ---
    df['driver_form_momentum'] = df['driver_pts_rolling3'] - df['driver_pts_rolling10']

    # --- DNF tendency ---
    df['driver_dnf_rolling5'] = (
        grp['DNF']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    # --- Qualifying pace ---
    df['driver_quali_rolling3'] = (
        grp['qualifying_pos']
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    df['driver_quali_ewm3'] = (
        grp['qualifying_pos']
        .transform(lambda x: x.shift(1).ewm(span=3, min_periods=1).mean())
    )

    # --- Experience ---
    df['driver_race_count'] = grp.cumcount()

    # --- Consistency (volatility of points) ---
    df['driver_pts_std_rolling5'] = (
        grp['fantasy_points_total']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=2).std())
    )

    # =========================================================
    # CONSTRUCTOR ROLLING FEATURES
    # =========================================================
    df = df.sort_values(['constructor_norm', 'race_idx']).reset_index(drop=True)
    con_grp = df.groupby('constructor_norm')

    df['constructor_pts_rolling3'] = (
        con_grp['fantasy_points_total']
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    df['constructor_pts_ewm5'] = (
        con_grp['fantasy_points_total']
        .transform(lambda x: x.shift(1).ewm(span=5, min_periods=1).mean())
    )

    # =========================================================
    # TEAMMATE COMPARISON
    # Controls for car pace; isolates driver skill signal.
    # =========================================================
    df = df.sort_values(['season', 'round', 'driver_id']).reset_index(drop=True)
    race_con_grp = df.groupby(['season', 'round', 'constructor_norm'])

    # Qualifying delta (how much better/worse than teammate in quali)
    df['teammate_quali_avg'] = race_con_grp['qualifying_pos'].transform('mean')
    df['quali_delta_from_teammate'] = df['qualifying_pos'] - df['teammate_quali_avg']

    # Race points delta vs teammate — rolling 5-race history (no leakage)
    df['_teammate_pts_avg'] = race_con_grp['fantasy_points_total'].transform('mean')
    df['_pts_vs_teammate']  = df['fantasy_points_total'] - df['_teammate_pts_avg']
    df = df.sort_values(['driver_id', 'race_idx']).reset_index(drop=True)
    grp = df.groupby('driver_id')
    df['teammate_pts_delta_rolling5'] = (
        grp['_pts_vs_teammate']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    # =========================================================
    # CIRCUIT-SPECIFIC HISTORY
    # =========================================================
    # Driver avg at this specific circuit (all prior visits)
    df = df.sort_values(['driver_id', 'event_name', 'race_idx']).reset_index(drop=True)
    circ_grp = df.groupby(['driver_id', 'event_name'])
    df['circuit_driver_avg'] = (
        circ_grp['fantasy_points_total']
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    # Circuit baseline: all drivers, all prior visits
    df = df.sort_values(['event_name', 'race_idx']).reset_index(drop=True)
    circ_all_grp = df.groupby('event_name')
    df['circuit_avg_pts'] = (
        circ_all_grp['fantasy_points_total']
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    df['circuit_dnf_rate'] = (
        circ_all_grp['DNF']
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    # =========================================================
    # CALENDAR
    # =========================================================
    season_lengths = df.groupby('season')['round'].transform('max')
    df['round_pct'] = df['round'] / season_lengths

    # =========================================================
    # DRIVER PROFILE FEATURES
    # =========================================================

    # --- Age at race time (season year used as proxy) ---
    df['driver_age'] = df.apply(
        lambda r: r['season'] - DRIVER_BIRTH_YEAR.get(r['driver_id'], np.nan),
        axis=1
    )

    # --- Years in F1 prior to this season ---
    # Computed from our own data: count distinct seasons strictly before
    # the current season in which this driver_id appears.
    driver_season_set = df.groupby('driver_id')['season'].apply(set).to_dict()
    df['years_in_f1'] = df.apply(
        lambda r: len([s for s in driver_season_set.get(r['driver_id'], set())
                       if s < r['season']]),
        axis=1
    )

    # --- Team change flag ---
    # 1 if the driver's constructor changed since their previous race, else 0.
    # First race for any driver defaults to 1 (joining the grid = "new team").
    df = df.sort_values(['driver_id', 'race_idx']).reset_index(drop=True)
    df['_prev_constructor'] = df.groupby('driver_id')['constructor_norm'].shift(1)
    df['new_team_flag'] = (
        (df['constructor_norm'] != df['_prev_constructor'])
        .astype(int)
        .where(df['_prev_constructor'].notna(), other=1)
    )

    # =========================================================
    # SEASON-TO-DATE FEATURES  (no-leakage: cumsum shifted by 1 round)
    # =========================================================

    # --- Driver season pts so far (cumulative in current season) ---
    df = df.sort_values(['driver_id', 'season', 'round']).reset_index(drop=True)
    df['driver_season_pts_so_far'] = (
        df.groupby(['driver_id', 'season'])['fantasy_points_total']
        .transform(lambda x: x.shift(1).cumsum().fillna(0))
    )

    # --- Constructor season pts so far (sum of both drivers, cumulative) ---
    # Aggregate to constructor-round level first to avoid double-counting,
    # then expand back to driver rows.
    df = df.sort_values(['constructor_norm', 'season', 'round']).reset_index(drop=True)
    con_round = (
        df.groupby(['constructor_norm', 'season', 'round'])['fantasy_points_total']
        .sum()
        .reset_index()
        .rename(columns={'fantasy_points_total': '_con_round_total'})
    )
    con_round = con_round.sort_values(['constructor_norm', 'season', 'round'])
    con_round['constructor_season_pts_so_far'] = (
        con_round.groupby(['constructor_norm', 'season'])['_con_round_total']
        .transform(lambda x: x.shift(1).cumsum().fillna(0))
    )
    df = df.merge(
        con_round[['constructor_norm', 'season', 'round', 'constructor_season_pts_so_far']],
        on=['constructor_norm', 'season', 'round'], how='left'
    )

    # =========================================================
    # QUALIFYING SESSION REACHED  (derived from qualifying_pos — no new data)
    # Q3 = top 10, Q2 = 11–15, Q1 = 16+, 0 = no time / DNS
    # =========================================================
    def _quali_session(pos):
        if pd.isna(pos):
            return 0
        p = int(pos)
        if p <= 10:
            return 3
        if p <= 15:
            return 2
        return 1

    df['quali_session_reached'] = df['qualifying_pos'].apply(_quali_session)

    # =========================================================
    # WET WEATHER SPECIALIST
    # Expanding mean of fantasy pts in wet races (rainfall_avg > 0.3)
    # shifted to avoid leakage. Drivers with no prior wet races → NaN.
    # =========================================================
    df = df.sort_values(['driver_id', 'race_idx']).reset_index(drop=True)
    grp = df.groupby('driver_id')

    # Tag each race as wet (True) or not; mask points to NaN for dry races
    df['_wet_race'] = (df['rainfall_avg'].fillna(0) > 0.3).astype(float)
    df['_wet_pts']  = df['fantasy_points_total'].where(df['_wet_race'] == 1)

    df['driver_wet_pts_avg'] = (
        grp['_wet_pts']
        .transform(lambda x: x.shift(1).expanding().mean())
    )
    # Interaction: wet specialist advantage × current race rainfall
    df['wet_specialist_score'] = df['driver_wet_pts_avg'] * df['rainfall_avg'].fillna(0)

    # =========================================================
    # FASTEST LAP RATE
    # Rolling rate (last 10 races) at which this driver has achieved fastest lap.
    # Drivers at the front on raw pace get fastest lap far more often (VER 21%,
    # HAM 18.5% vs midfield <3%), so this adds signal that qualifying_pos alone
    # can't fully capture (teams sometimes pit for a fresh tyre to attempt FL).
    # =========================================================
    df = df.sort_values(['driver_id', 'race_idx']).reset_index(drop=True)
    grp = df.groupby('driver_id')
    if 'fastest_lap' in df.columns:
        df['_fl_flag'] = df['fastest_lap'].fillna(False).astype(float)
    else:
        df['_fl_flag'] = 0.0
    df['driver_fl_rate_rolling10'] = (
        grp['_fl_flag']
        .transform(lambda x: x.shift(1).rolling(10, min_periods=1).mean())
    )

    # =========================================================
    # PRACTICE SESSION PACE  (current-weekend feature — no leakage)
    # fp2_gap_pct is most representative (race setup, full fuel);
    # fp1_gap_pct is an earlier signal on longer runs.
    # These columns come from DataRetrieval/practice.py and are stored
    # in the pickles. Older rounds without practice data are NaN
    # and handled by the imputer.
    # =========================================================
    # Columns already in df from pickles — just ensure they exist
    for col in ['fp1_gap_pct', 'fp2_gap_pct', 'fp3_gap_pct']:
        if col not in df.columns:
            df[col] = float('nan')

    # =========================================================
    # CIRCUIT TYPE FLAGS
    # =========================================================
    df['is_street_circuit']   = df['event_name'].isin(STREET_CIRCUITS).astype(int)
    df['is_high_altitude']    = df['event_name'].isin(HIGH_ALTITUDE_CIRCUITS).astype(int)

    df = df.sort_values(['season', 'round', 'driver_id']).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 3. FEATURE SELECTION
#    Only features available BEFORE the race starts.
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    # Pre-race knowns
    'qualifying_pos',
    'grid',
    'had_sprint',
    'round_pct',

    # Driver form — multiple timescales + exponential weighting
    'driver_pts_rolling3',
    'driver_pts_rolling5',
    'driver_pts_rolling10',
    'driver_pts_ewm5',
    'driver_form_momentum',
    'driver_dnf_rolling5',
    'driver_quali_rolling3',
    'driver_quali_ewm3',
    'driver_race_count',
    'driver_pts_std_rolling5',

    # Teammate comparison (isolates driver skill from car pace)
    'quali_delta_from_teammate',
    'teammate_pts_delta_rolling5',

    # Constructor form
    'constructor_pts_rolling3',
    'constructor_pts_ewm5',

    # Circuit-specific history
    'circuit_driver_avg',
    'circuit_avg_pts',
    'circuit_dnf_rate',

    # Weather (available as a forecast pre-race)
    'track_temp_avg',
    'air_temp_avg',
    'humidity_avg',
    'rainfall_avg',
    'wind_speed_avg',

    # Driver profile
    'driver_age',
    'years_in_f1',
    'new_team_flag',

    # Season-to-date (dynamic car quality + in-season form)
    'driver_season_pts_so_far',
    'constructor_season_pts_so_far',

    # Circuit type
    'is_street_circuit',
    'is_high_altitude',

    # Qualifying session reached (Q1 / Q2 / Q3 — sharper than raw position)
    'quali_session_reached',

    # Wet weather specialist
    'driver_wet_pts_avg',
    'wet_specialist_score',

    # Practice session pace (current weekend, available before the race)
    'fp1_gap_pct',
    'fp2_gap_pct',

    # Fastest lap tendency (rolling 10-race rate — top drivers 15-21%, midfield <3%)
    'driver_fl_rate_rolling10',
]

TARGET_COL = 'fantasy_points_total'


# ---------------------------------------------------------------------------
# 4. PIPELINES
#    Tree boosters (XGB, LGBM) handle NaN natively — no imputer/scaler.
#    Sklearn models (RF, GBR) need imputation.
# ---------------------------------------------------------------------------

def build_sklearn_pipeline(model):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('model',   model),
    ])


def build_boosting_pipeline(model):
    """XGBoost / LightGBM — NaN-safe, scale-invariant."""
    return Pipeline([('model', model)])


# ---------------------------------------------------------------------------
# 5. TRAIN & EVALUATE
# ---------------------------------------------------------------------------

def train_and_evaluate(df: pd.DataFrame, eval_only=False):
    df = engineer_features(df)

    train_mask = df['season'].isin(TRAIN_SEASONS)
    test_mask  = df['season'].isin(TEST_SEASONS)

    X_train = df.loc[train_mask, FEATURE_COLS]
    y_train = df.loc[train_mask, TARGET_COL]
    X_test  = df.loc[test_mask,  FEATURE_COLS]
    y_test  = df.loc[test_mask,  TARGET_COL]

    print(f"Train: {train_mask.sum()} rows  ({TRAIN_SEASONS})")
    print(f"Test:  {test_mask.sum()} rows   ({TEST_SEASONS})")
    print(f"Features: {len(FEATURE_COLS)}  (baseline had 20)")
    print()

    models = {
        'RandomForest': build_sklearn_pipeline(
            RandomForestRegressor(
                n_estimators=400, max_depth=10, min_samples_leaf=4,
                random_state=42, n_jobs=-1
            )
        ),
        'XGBoost': build_boosting_pipeline(
            XGBRegressor(
                n_estimators=500, learning_rate=0.04, max_depth=5,
                subsample=0.8, colsample_bytree=0.8,
                min_child_weight=5, reg_lambda=2.0,
                random_state=42, n_jobs=-1,
                verbosity=0, eval_metric='mae',
            )
        ),
        'LightGBM': build_boosting_pipeline(
            LGBMRegressor(
                n_estimators=500, learning_rate=0.04, max_depth=6,
                num_leaves=40, subsample=0.8, colsample_bytree=0.8,
                min_child_samples=10, reg_lambda=2.0,
                random_state=42, n_jobs=-1, verbose=-1,
            )
        ),
    }

    results = {}
    report_lines = [
        f"Baseline (v1 RandomForest): MAE={BASELINE_MAE:.2f}",
        "",
    ]
    print(f"{'─'*72}")

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        mae     = mean_absolute_error(y_test, preds)
        r2      = r2_score(y_test, preds)
        within5  = (np.abs(preds - y_test) <= 5).mean()
        within10 = (np.abs(preds - y_test) <= 10).mean()
        delta    = BASELINE_MAE - mae   # positive = better than baseline

        results[name] = {'pipe': pipe, 'mae': mae, 'r2': r2}

        line = (f"{name:<14}  MAE={mae:.2f} ({delta:+.2f} vs baseline)"
                f"  R²={r2:.3f}"
                f"  ±5pts={within5:.0%}  ±10pts={within10:.0%}")
        print(line)
        report_lines.append(line)

    print(f"{'─'*72}")

    # --- Pick best by MAE ---
    best_name  = min(results, key=lambda k: results[k]['mae'])
    best_pipe  = results[best_name]['pipe']
    best_model = best_pipe['model']

    print(f"\nBest model: {best_name} (MAE={results[best_name]['mae']:.2f})")

    if hasattr(best_model, 'feature_importances_'):
        importances = pd.Series(best_model.feature_importances_, index=FEATURE_COLS)
        importances = importances.sort_values(ascending=False)
        print(f"\nFeature importances ({best_name}):")
        report_lines.append(f"\nFeature importances ({best_name}):")
        for feat, imp in importances.items():
            line = f"  {feat:<32} {imp:.4f}"
            print(line)
            report_lines.append(line)

    # --- Ranking accuracy (Precision@K) using the best model ---
    test_df = df[test_mask].copy()
    test_df['predicted'] = best_pipe.predict(X_test)
    test_df['actual_rank']    = test_df.groupby(['season', 'round'])['fantasy_points_total'].rank(ascending=False)
    test_df['predicted_rank'] = test_df.groupby(['season', 'round'])['predicted'].rank(ascending=False)

    def precision_at_k(group, k=3):
        actual_top    = set(group.nsmallest(k, 'actual_rank')['driver_id'])
        predicted_top = set(group.nsmallest(k, 'predicted_rank')['driver_id'])
        return len(actual_top & predicted_top) / k

    p3 = test_df.groupby(['season', 'round']).apply(precision_at_k, k=3).mean()
    p5 = test_df.groupby(['season', 'round']).apply(precision_at_k, k=5).mean()

    line3 = f"\nPrecision@3 (overlap of predicted/actual top 3): {p3:.0%}  (baseline: 56%)"
    line5 = f"Precision@5 (overlap of predicted/actual top 5): {p5:.0%}  (baseline: 70%)"
    print(line3); print(line5)
    report_lines.extend([line3, line5])

    if not eval_only:
        os.makedirs(MODELS_DIR, exist_ok=True)

        # Retrain on train+test combined before saving
        X_all = df.loc[train_mask | test_mask, FEATURE_COLS]
        y_all = df.loc[train_mask | test_mask, TARGET_COL]
        best_pipe.fit(X_all, y_all)

        joblib.dump(best_pipe, os.path.join(MODELS_DIR, 'fantasy_model.joblib'))
        with open(os.path.join(MODELS_DIR, 'feature_columns.json'), 'w') as f:
            json.dump(FEATURE_COLS, f, indent=2)
        with open(os.path.join(MODELS_DIR, 'model_eval.txt'), 'w') as f:
            f.write('\n'.join(report_lines))

        print(f"\nSaved: {MODELS_DIR}/fantasy_model.joblib  ({best_name})")
        print(f"Saved: {MODELS_DIR}/feature_columns.json")
        print(f"Saved: {MODELS_DIR}/model_eval.txt")

    return best_pipe, results


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    eval_only = '--eval-only' in sys.argv

    print("Loading data...")
    df = load_all_data()
    print(f"  {len(df)} rows across {df['season'].nunique()} seasons\n")

    train_and_evaluate(df, eval_only=eval_only)

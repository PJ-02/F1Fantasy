"""
Trains a regression model to predict driver fantasy_points_total for an upcoming race.
Only features available BEFORE the race starts are used (qualifying results + rolling history).

Outputs:
  models/fantasy_model.joblib       — trained pipeline (imputer + scaler + GBR)
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

warnings.filterwarnings('ignore')

CACHE_DIR   = 'cache/pickles'
MODELS_DIR  = 'models'
TRAIN_SEASONS = [2019, 2020, 2021, 2022, 2023]
TEST_SEASONS  = [2024]
# 2025/2026 held out entirely — used only for live prediction


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

    # Numeric DNF flag (already computed in preprocess, ensure dtype)
    df['DNF'] = df['DNF'].astype(int)

    # --- Race index for ordering (avoids sort issues) ---
    df['race_idx'] = df['season'] * 100 + df['round']

    # === Driver rolling features ===
    # Sorted by race_idx within each driver group
    df = df.sort_values(['driver_id', 'race_idx']).reset_index(drop=True)

    grp = df.groupby('driver_id')

    df['driver_pts_rolling3'] = (
        grp['fantasy_points_total']
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    df['driver_pts_rolling5'] = (
        grp['fantasy_points_total']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    df['driver_dnf_rolling5'] = (
        grp['DNF']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    df['driver_quali_rolling3'] = (
        grp['qualifying_pos']
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    # How many races this driver has done — proxy for "experience"
    df['driver_race_count'] = grp.cumcount()

    # === Constructor rolling features ===
    df = df.sort_values(['constructor_norm', 'race_idx']).reset_index(drop=True)
    con_grp = df.groupby('constructor_norm')
    df['constructor_pts_rolling3'] = (
        con_grp['fantasy_points_total']
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    # === Teammate comparison: quali delta vs teammate ===
    # Controls for car pace; positive = worse than teammate, negative = better
    df = df.sort_values(['season', 'round', 'driver_id']).reset_index(drop=True)
    race_constructor_grp = df.groupby(['season', 'round', 'constructor_norm'])
    df['teammate_quali_avg'] = race_constructor_grp['qualifying_pos'].transform('mean')
    df['quali_delta_from_teammate'] = df['qualifying_pos'] - df['teammate_quali_avg']

    # === Driver consistency: std dev of pts over last 5 races ===
    df = df.sort_values(['driver_id', 'race_idx']).reset_index(drop=True)
    grp = df.groupby('driver_id')
    df['driver_pts_std_rolling5'] = (
        grp['fantasy_points_total']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=2).std())
    )

    # === Circuit-specific: driver avg at this circuit (all prior visits) ===
    # cumulative mean over previous appearances at the same (driver, circuit) pair
    df = df.sort_values(['driver_id', 'event_name', 'race_idx']).reset_index(drop=True)
    circ_grp = df.groupby(['driver_id', 'event_name'])
    df['circuit_driver_avg'] = (
        circ_grp['fantasy_points_total']
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    # Circuit baseline: overall avg fantasy points at this circuit (all drivers, all prior races)
    # computed as expanding mean sorted by race_idx
    df = df.sort_values(['event_name', 'race_idx']).reset_index(drop=True)
    circ_all_grp = df.groupby('event_name')
    df['circuit_avg_pts'] = (
        circ_all_grp['fantasy_points_total']
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    # Circuit DNF rate: rolling historical DNF rate at this circuit (prior editions)
    df['circuit_dnf_rate'] = (
        circ_all_grp['DNF']
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    # === Season progress ===
    season_lengths = df.groupby('season')['round'].transform('max')
    df['round_pct'] = df['round'] / season_lengths   # 0→1 through the season

    # Re-sort chronologically
    df = df.sort_values(['season', 'round', 'driver_id']).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# 3. FEATURE SELECTION
#    Only features available BEFORE the race:
#      - qualifying_pos / grid (available after qualifying)
#      - rolling driver/constructor history
#      - circuit history
#      - weather forecast
#      - calendar info (sprint flag, round position in season)
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    # Pre-race known
    'qualifying_pos',
    'grid',
    'had_sprint',
    'round_pct',

    # Driver form (rolling history)
    'driver_pts_rolling3',
    'driver_pts_rolling5',
    'driver_dnf_rolling5',
    'driver_quali_rolling3',
    'driver_race_count',
    'driver_pts_std_rolling5',

    # Teammate comparison (controls for car pace)
    'quali_delta_from_teammate',

    # Constructor form
    'constructor_pts_rolling3',

    # Circuit-specific history
    'circuit_driver_avg',
    'circuit_avg_pts',
    'circuit_dnf_rate',

    # Weather (available as forecast)
    'track_temp_avg',
    'air_temp_avg',
    'humidity_avg',
    'rainfall_avg',
    'wind_speed_avg',
]

TARGET_COL = 'fantasy_points_total'


# ---------------------------------------------------------------------------
# 4. BUILD SKLEARN PIPELINE
# ---------------------------------------------------------------------------

def build_pipeline(model):
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
        ('model',   model),
    ])


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

    print(f"Train: {train_mask.sum()} rows ({TRAIN_SEASONS})")
    print(f"Test:  {test_mask.sum()} rows  ({TEST_SEASONS})")
    print(f"Features: {len(FEATURE_COLS)}")
    print()

    models = {
        'RandomForest': build_pipeline(RandomForestRegressor(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            random_state=42, n_jobs=-1
        )),
        'GradientBoosting': build_pipeline(GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=4,
            min_samples_leaf=5, subsample=0.8, random_state=42
        )),
    }

    results = {}
    report_lines = []

    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        mae  = mean_absolute_error(y_test, preds)
        r2   = r2_score(y_test, preds)
        within5  = (np.abs(preds - y_test) <= 5).mean()
        within10 = (np.abs(preds - y_test) <= 10).mean()

        results[name] = {'pipe': pipe, 'mae': mae, 'r2': r2}

        line = (f"{name:<22}  MAE={mae:.2f}  R²={r2:.3f}"
                f"  within±5pts={within5:.0%}  within±10pts={within10:.0%}")
        print(line)
        report_lines.append(line)

    # Feature importance from the better model (GBR)
    gbr_model = models['GradientBoosting']['model']
    importances = pd.Series(gbr_model.feature_importances_, index=FEATURE_COLS)
    importances = importances.sort_values(ascending=False)

    print()
    print("Feature importances (GradientBoosting):")
    report_lines.append("\nFeature importances (GradientBoosting):")
    for feat, imp in importances.items():
        line = f"  {feat:<30} {imp:.4f}"
        print(line)
        report_lines.append(line)

    # Per-position accuracy on test set (how well we predict top finishers)
    test_df = df[test_mask].copy()
    test_df['predicted'] = models['GradientBoosting'].predict(X_test)
    test_df['actual_rank']    = test_df.groupby(['season','round'])['fantasy_points_total'].rank(ascending=False)
    test_df['predicted_rank'] = test_df.groupby(['season','round'])['predicted'].rank(ascending=False)

    # Precision@3: how often does our predicted top-3 overlap with actual top-3?
    def precision_at_k(group, k=3):
        actual_top    = set(group.nsmallest(k, 'actual_rank')['driver_id'])
        predicted_top = set(group.nsmallest(k, 'predicted_rank')['driver_id'])
        return len(actual_top & predicted_top) / k

    p3 = test_df.groupby(['season','round']).apply(precision_at_k, k=3).mean()
    p5 = test_df.groupby(['season','round']).apply(precision_at_k, k=5).mean()

    line3 = f"\nPrecision@3 (overlap of predicted/actual top 3): {p3:.0%}"
    line5 = f"Precision@5 (overlap of predicted/actual top 5): {p5:.0%}"
    print(line3); print(line5)
    report_lines.extend([line3, line5])

    # Pick the best model by MAE
    best_name = min(results, key=lambda k: results[k]['mae'])
    best_pipe  = results[best_name]['pipe']
    print(f"\nBest model: {best_name} (MAE={results[best_name]['mae']:.2f})")

    if not eval_only:
        os.makedirs(MODELS_DIR, exist_ok=True)

        # Retrain best model on train+test combined before saving
        X_all = df.loc[train_mask | test_mask, FEATURE_COLS]
        y_all = df.loc[train_mask | test_mask, TARGET_COL]
        best_pipe.fit(X_all, y_all)

        joblib.dump(best_pipe, os.path.join(MODELS_DIR, 'fantasy_model.joblib'))
        with open(os.path.join(MODELS_DIR, 'feature_columns.json'), 'w') as f:
            json.dump(FEATURE_COLS, f, indent=2)
        with open(os.path.join(MODELS_DIR, 'model_eval.txt'), 'w') as f:
            f.write('\n'.join(report_lines))

        print(f"Saved: {MODELS_DIR}/fantasy_model.joblib")
        print(f"Saved: {MODELS_DIR}/feature_columns.json")
        print(f"Saved: {MODELS_DIR}/model_eval.txt")

    return best_pipe, importances


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    eval_only = '--eval-only' in sys.argv

    print("Loading data...")
    df = load_all_data()
    print(f"  {len(df)} rows across {df['season'].nunique()} seasons\n")

    train_and_evaluate(df, eval_only=eval_only)

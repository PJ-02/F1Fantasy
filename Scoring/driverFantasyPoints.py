from Scoring.preprocess import _is_dnf, _is_dsq

def calculate_fantasy_points(df):
    # --- Qualifying Points ---
    quali_points_map = {1: 10, 2: 9, 3: 8, 4: 7, 5: 6, 6: 5, 7: 4, 8: 3, 9: 2, 10: 1}
    df['qualifying_points'] = df['qualifying_pos'].map(quali_points_map).fillna(0)

    # -5 for no time set (NaN position, not DSQ)
    df.loc[(df['qualifying_pos'].isna()) & (df['qualifying_disqualified'] == 0), 'qualifying_points'] = -5
    # -5 for DSQ
    df.loc[df['qualifying_disqualified'] == 1, 'qualifying_points'] = -5


    # --- Sprint Points ---
    sprint_points_map = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}
    df['sprint_points'] = df['sprint_pos'].map(sprint_points_map).fillna(0)

    # Positions Gained / Lost
    df['sprint_gain_points'] = df['sprint_positions_gained'].fillna(0)

    # Fastest lap in sprint
    if 'sprint_fastest_lap' in df.columns:
        df['sprint_fastest_lap_bonus'] = df['sprint_fastest_lap'].apply(lambda x: 5 if x else 0)
    else:
        df['sprint_fastest_lap_bonus'] = 0

    # Sprint Overtakes
    if 'sprint_overtakes' not in df.columns:
        df['sprint_overtakes'] = 0
    df['sprint_overtake_points'] = df['sprint_overtakes'].fillna(0).astype(int)

    # DNF/NC penalty: -10 per 2026 rules
    df['sprint_dnf_penalty'] = df['sprint_status'].apply(
        lambda x: -10 if _is_dnf(x) else 0
    )
    # DSQ penalty: -10 per 2026 rules
    df['sprint_dsq_penalty'] = df['sprint_disqualified'] * -10

    # Total sprint score
    df['sprint_total'] = (
        df['sprint_points'] +
        df['sprint_gain_points'] +
        df['sprint_overtake_points'] +
        df['sprint_fastest_lap_bonus'] +
        df['sprint_dnf_penalty'] +
        df['sprint_dsq_penalty']
    )


    # --- Race Points ---
    race_points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
    df['race_points'] = df['position'].map(race_points_map).fillna(0)

    # Positions Gained / Lost
    df['race_gain_points'] = df['positions_gained'].fillna(0)

    # Fastest Lap
    df['race_fastest_lap_bonus'] = df['fastest_lap'].apply(lambda x: 10 if x else 0)

    # Driver of the Day — sourced externally; defaulting to 0 until real data is available
    df['dotd_bonus'] = 0

    # Overtakes
    if 'overtakes' not in df.columns:
        df['overtakes'] = 0
    df['race_overtake_points'] = df['overtakes'].fillna(0).astype(int)

    # DNF / NC penalty: -20
    df['race_dnf_penalty'] = df['status'].apply(
        lambda x: -20 if _is_dnf(x) or _is_dsq(x) else 0
    )


    # --- Total Fantasy Score ---
    df['fantasy_points_total'] = (
        df['qualifying_points'] +
        df['sprint_total'] +
        df['race_points'] +
        df['race_gain_points'] +
        df['race_overtake_points'] +
        df['race_fastest_lap_bonus'] +
        df['dotd_bonus'] +
        df['race_dnf_penalty']
    )

    return df
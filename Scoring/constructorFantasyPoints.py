import pandas as pd

from Scoring.preprocess import _is_dsq

def calculate_constructor_points(driver_df, pitstop_df):
    # Detect fastest pitstop of the race
    if pitstop_df is None or pitstop_df.empty or 'fastest_pitstop_time' not in pitstop_df.columns:
        fastest_team = None
    else:
        fastest_pitstop = pitstop_df['fastest_pitstop_time'].min()
        fastest_team_rows = pitstop_df.loc[pitstop_df['fastest_pitstop_time'] == fastest_pitstop, 'constructor_name']
        fastest_team = fastest_team_rows.iloc[0] if not fastest_team_rows.empty else None


    # World record benchmark
    WORLD_RECORD_TIME = 1.80

    # Group by constructor
    grouped = driver_df.groupby('constructor_name')

    constructor_rows = []

    for constructor, group in grouped:
        entry = {'constructor_name': constructor}

        # Combine driver-level fantasy points
        entry['driver_points_total'] = group['fantasy_points_total'].sum()

        # Qualifying Bonuses
        q2_count = group['qualifying_pos'].apply(lambda x: x <= 15 if pd.notna(x) else False).sum()
        q3_count = group['qualifying_pos'].apply(lambda x: x <= 10 if pd.notna(x) else False).sum()

        if q2_count == 0:
            qual_bonus = -1
        elif q2_count == 1:
            qual_bonus = 1
        else:
            qual_bonus = 3

        if q3_count == 1:
            qual_bonus += 5
        elif q3_count == 2:
            qual_bonus += 10

        entry['qualifying_bonus'] = qual_bonus

        # Pitstop time from FastF1
        pitstop_points = 0
        if pitstop_df is not None and constructor in pitstop_df['constructor_name'].values:
            p_time = pitstop_df.loc[pitstop_df['constructor_name'] == constructor, 'fastest_pitstop_time'].values[0]
            entry['fastest_pitstop_time'] = p_time

            if p_time < 2.0:
                pitstop_points = 20
            elif 2.0 <= p_time < 2.2:
                pitstop_points = 10
            elif 2.2 <= p_time < 2.5:
                pitstop_points = 5
            elif 2.5 <= p_time < 3.0:
                pitstop_points = 2
            else:
                pitstop_points = 0
        else:
            entry['fastest_pitstop_time'] = None
        entry['pitstop_points'] = pitstop_points

        entry['fastest_pitstop_bonus'] = 0  # Always initialize first

        if constructor == fastest_team:
            entry['fastest_pitstop_bonus'] += 5
            team_pitstop_time = pitstop_df.loc[pitstop_df['constructor_name'] == constructor, 'fastest_pitstop_time'].values[0]
            if team_pitstop_time < WORLD_RECORD_TIME:
                entry['fastest_pitstop_bonus'] += 15


        # Constructor penalties — sprint DSQ: -10/driver, race DSQ: -20/driver (2026 rules)
        sprint_dsq_count = group['sprint_disqualified'].sum()
        race_dsq_count = group['status'].apply(lambda x: 1 if _is_dsq(x) else 0).sum()

        entry['disqualification_penalty'] = (sprint_dsq_count * -10) + (race_dsq_count * -20)


        # Final Constructor Fantasy Points
        entry['constructor_fantasy_points'] = (
        entry['driver_points_total'] +
        entry['qualifying_bonus'] +
        entry['pitstop_points'] +
        entry['fastest_pitstop_bonus'] +
        entry['disqualification_penalty']
        )


        constructor_rows.append(entry)

    return pd.DataFrame(constructor_rows)
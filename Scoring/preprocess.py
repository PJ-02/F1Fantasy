_DNF_STATUSES = {'accident', 'retired', 'not classified', 'nc', 'dnf', 'collision', 'mechanical', 'power unit', 'gearbox', 'hydraulics', 'brakes', 'engine', 'suspension', 'electrical', 'spun off', 'withdrew', 'did not finish'}
_DSQ_STATUSES = {'disqualified', 'dsq', 'excluded'}

def _is_dnf(status):
    return any(s in str(status).lower() for s in _DNF_STATUSES)

def _is_dsq(status):
    return any(s in str(status).lower() for s in _DSQ_STATUSES)

def preprocess_driver_data(df):
    df['DNF'] = df['status'].apply(lambda x: 1 if _is_dnf(x) else 0)
    df['positions_gained'] = df['grid'] - df['position']
    df['positions_gained'] = df['positions_gained'].fillna(0)

    # Sprint fantasy features
    df['sprint_positions_gained'] = df['sprint_grid'] - df['sprint_pos']
    df['sprint_positions_gained'] = df['sprint_positions_gained'].fillna(0).astype(int)
    df['sprint_dnf'] = df['sprint_status'].apply(lambda x: 1 if _is_dnf(x) else 0)
    df['sprint_disqualified'] = df['sprint_status'].apply(lambda x: 1 if _is_dsq(x) else 0)

    # Use qualifying_status (from FastF1 Status field) for DSQ detection, not qualifying_pos
    if 'qualifying_status' in df.columns:
        df['qualifying_disqualified'] = df['qualifying_status'].apply(lambda x: 1 if _is_dsq(x) else 0)
    else:
        df['qualifying_disqualified'] = 0

    return df
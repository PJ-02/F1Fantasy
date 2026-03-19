"""
Fetch true stationary pit stop times from the OpenF1 API.

OpenF1 provides a `stop_duration` field (the time the car is stationary in the
box, ~2–4s) that matches the F1 Fantasy scoring thresholds.  This field has been
populated since the 2024 United States Grand Prix (Austin, Round 19 of 2024).

For earlier sessions `stop_duration` is null and this module returns an empty
DataFrame, signalling the caller to fall back to FastF1 traversal times.

OpenF1 endpoints used:
  GET /v1/sessions?year={year}&round_number={round_no}&session_type=Race
  GET /v1/pit?session_key={key}
  GET /v1/drivers?session_key={key}
"""

import time
import requests
import pandas as pd

from Utils.common import logger

_OPENF1_BASE = "https://api.openf1.org/v1"
_TIMEOUT = 15  # seconds per request
_DELAY = 1.0   # seconds between API calls to avoid 429s

# Cache the session list per year so repeated round calls don't re-fetch it
_SESSION_CACHE: dict[int, list] = {}


def _get(endpoint: str, params: dict) -> list:
    url = f"{_OPENF1_BASE}/{endpoint}"
    try:
        r = requests.get(url, params=params, timeout=_TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"OpenF1 request failed ({url} {params}): {e}")
        return []


def _get_race_sessions(year: int) -> list:
    """Cached fetch of all Race sessions for a year."""
    if year not in _SESSION_CACHE:
        sessions = _get("sessions", {"year": year, "session_type": "Race", "session_name": "Race"})
        sessions.sort(key=lambda s: s.get("date_start", ""))
        _SESSION_CACHE[year] = sessions
        time.sleep(_DELAY)
    return _SESSION_CACHE[year]


def get_openf1_pitstops(season: int, round_no: int) -> pd.DataFrame:
    """
    Return a DataFrame with columns [constructor_name, fastest_pitstop_time]
    where fastest_pitstop_time is the true stationary stop duration in seconds.

    Returns an empty DataFrame if:
    - The session cannot be found in OpenF1
    - stop_duration data is unavailable (pre-Austin 2024 sessions)
    - Any network/API error occurs
    """
    # 1. Look up the Race session key
    # OpenF1 does not expose round_number; fetch all Race sessions for the year
    # sorted chronologically — the Nth session (0-indexed) is round N.
    # Filter by session_name="Race" to exclude Sprint sessions which also have
    # session_type="Race" but are a separate event per weekend.
    all_sessions = _get_race_sessions(season)
    if not all_sessions:
        logger.debug(f"OpenF1: no Race sessions found for {season}")
        return pd.DataFrame(columns=['constructor_name', 'fastest_pitstop_time'])

    all_sessions.sort(key=lambda s: s.get("date_start", ""))
    idx = round_no - 1
    if idx < 0 or idx >= len(all_sessions):
        logger.debug(f"OpenF1: round {round_no} out of range (found {len(all_sessions)} sessions for {season})")
        return pd.DataFrame(columns=['constructor_name', 'fastest_pitstop_time'])

    session_key = all_sessions[idx].get("session_key")
    if not session_key:
        return pd.DataFrame(columns=['constructor_name', 'fastest_pitstop_time'])

    # 2. Fetch pit stop records
    pit_records = _get("pit", {"session_key": session_key})
    time.sleep(_DELAY)
    if not pit_records:
        logger.debug(f"OpenF1: no pit records for session_key={session_key}")
        return pd.DataFrame(columns=['constructor_name', 'fastest_pitstop_time'])

    pit_df = pd.DataFrame(pit_records)

    # Check if stop_duration exists and has usable values
    if "stop_duration" not in pit_df.columns:
        logger.debug(f"OpenF1: stop_duration column missing for {season} R{round_no}")
        return pd.DataFrame(columns=['constructor_name', 'fastest_pitstop_time'])

    pit_df = pit_df[pit_df["stop_duration"].notna()].copy()
    if pit_df.empty:
        logger.debug(f"OpenF1: all stop_duration values null for {season} R{round_no} — pre-Austin 2024 session")
        return pd.DataFrame(columns=['constructor_name', 'fastest_pitstop_time'])

    pit_df["stop_duration"] = pd.to_numeric(pit_df["stop_duration"], errors="coerce")
    pit_df = pit_df[pit_df["stop_duration"].between(1.0, 10.0)].copy()  # sanity filter
    if pit_df.empty:
        return pd.DataFrame(columns=['constructor_name', 'fastest_pitstop_time'])

    # 3. Fetch driver → team mapping for this session
    driver_records = _get("drivers", {"session_key": session_key})
    time.sleep(_DELAY)
    if not driver_records:
        logger.warning(f"OpenF1: could not fetch driver list for session_key={session_key}")
        return pd.DataFrame(columns=['constructor_name', 'fastest_pitstop_time'])

    driver_df = pd.DataFrame(driver_records)[["driver_number", "team_name"]].drop_duplicates("driver_number")

    # 4. Merge to attach team_name to each stop
    pit_df["driver_number"] = pd.to_numeric(pit_df["driver_number"], errors="coerce")
    driver_df["driver_number"] = pd.to_numeric(driver_df["driver_number"], errors="coerce")
    merged = pit_df.merge(driver_df, on="driver_number", how="left")

    if "team_name" not in merged.columns or merged["team_name"].isna().all():
        logger.warning(f"OpenF1: team_name unavailable for {season} R{round_no}")
        return pd.DataFrame(columns=['constructor_name', 'fastest_pitstop_time'])

    # 5. Fastest stationary stop per team
    team_fastest = (
        merged.groupby("team_name")["stop_duration"]
        .min()
        .reset_index()
        .rename(columns={"team_name": "constructor_name", "stop_duration": "fastest_pitstop_time"})
    )

    logger.info(
        f"OpenF1 pitstops {season} R{round_no}: "
        f"{len(merged)} stops across {len(team_fastest)} teams "
        f"(fastest {team_fastest['fastest_pitstop_time'].min():.3f}s)"
    )
    return team_fastest

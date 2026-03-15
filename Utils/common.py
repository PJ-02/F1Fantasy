import time
import random
import functools
import fastf1.req
import logging

logger = logging.getLogger(__name__)

def retry_on_rate_limit(max_retries=5, base_delay=60):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except fastf1.req.RateLimitExceededError:
                    delay = base_delay * (2 ** attempt) + random.randint(1, 10)
                    logger.warning(f"Rate limit hit. Retrying in {delay} seconds... (Attempt {attempt + 1})")
                    time.sleep(delay)
            logger.error(f"Max retries exceeded for {func.__name__}")
            return None
        return wrapper
    return decorator

@retry_on_rate_limit()
def get_session_with_retry(season, round_no, session_type):
    return fastf1.get_session(season, round_no, session_type)

@retry_on_rate_limit()
def load_session_with_retry(session, telemetry=True, laps=True):
    session.load(telemetry=telemetry, laps=laps)
    return session

def verify_session_data(session, expected_columns=None, context=""):
    if session.results is None or session.results.empty:
        logger.warning(f"[{context}] Empty or missing results.")
        return False

    num_drivers = len(session.results)
    if num_drivers < 18:
        logger.warning(f"[{context}] Unexpected number of drivers: {num_drivers}")

    if expected_columns:
        missing = [col for col in expected_columns if col not in session.results or session.results[col].isna().all()]
        if missing:
            logger.warning(f"[{context}] Missing or all-null columns: {missing}")

    if context == "Race":
        if 'PitInTime' not in session.laps.columns or session.laps['PitInTime'].isna().all():
            logger.warning(f"[{context}] Pit stop data is missing or empty.")

    logger.info(f"[{context}] Data verification complete.")
    return True

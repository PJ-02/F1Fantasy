---
name: F1Fantasy project state
description: Current state of the F1Fantasy project including data, completed steps, and what's next
type: project
---

Data pipeline is complete. Scoring bugs fixed. Ready for ML model (Step 2).

**Why:** Building an AI F1 Fantasy system — pipeline feeds ML model feeds team optimizer.

**How to apply:** When asked to build the ML model or team optimizer, this context explains what's already done.

## Data State
- 2019–2025: All rounds cached in `cache/pickles/` (151 rounds total)
- 2026: r1 (Australia) and r2 (China) cached
- Combined Excel files rebuilt in `GeneratedSpreadsheets/`
- `backfillTelemetry.py` run to fill in top_speed, avg_throttle, brake_usage, overtakes for all historical rounds
- All pickles rescored with corrected 2026 rules via `rescoreCache.py`

## Bugs Fixed (Steps 1 + data collection)
- Sprint DNF/DSQ penalty: -20 → -10
- Race DSQ now correctly gets -20 penalty (was 0)
- Constructor sprint/race DSQ penalties split correctly
- DOTD hardcode (Verstappen) removed — stubbed at 0
- qualifying_disqualified: now uses qualifying_status (FastF1 Status field) not position
- DNF/DSQ status detection expanded with comprehensive status list
- pick_fastest() None guard added to race.py and sprint.py
- telemetry.py: session.laps access guarded
- pitstops.py: accepts pre-loaded session, has try/except
- mainPipeline.py: session loaded once and passed to all functions; telemetry merged BEFORE scoring (fixes overtakes_x/y collision)
- df.get().fillna() crash fixed for overtakes columns
- mainPipeline.py: if __name__ == '__main__' guard added
- Duplicate verify_session_data removed from mainPipeline.py

## New Files
- `check_cache_coverage.py` — audits pickle cache for missing rounds + column quality
- `rescoreCache.py` — re-applies scoring rules to existing pickles without API calls
- `backfillTelemetry.py` — backfills telemetry from FastF1 disk cache into old pickles

## Known Remaining Issues
- Pitstop data: "No valid pitstop data" warning on many 2025/2026 rounds — pd.to_datetime fails on timedelta PitInTime/PitOutTime. Pre-existing bug. Constructor pitstop bonuses are 0 for affected rounds.
- DOTD: still stubbed at 0 (no FastF1 source for this)
- Telemetry unavailable for some 2026 rounds (FastF1 data not yet published for very recent races)

## Next Steps (priority order)
1. Fix pitstop timedelta bug in pitstops.py
2. Build ML model in trainModel.py (Step 2)
3. Build teamOptimizer.py (Step 3)
4. Build predictRace.py (Step 4)

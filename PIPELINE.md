# Daily Data Refresh Pipeline

This document is the operational guide for Claude (or Rob) to run and troubleshoot the incremental data pipeline that powers py-players.streamlit.app.

---

## How the Pipeline Works

The pipeline is fully orchestrated by `run_daily_refresh.py`. It:

1. Reads `data/logs/last_pull_date.txt` to determine what dates to pull
2. Pulls new pitch rows from the API for the current season only
3. Accumulates them into `data/raw/pitch_data_{season}.parquet` (deduplicating on `pa_id` + `pitch_of_ab`)
4. Re-aggregates the current season into `data/output/_season_chunks/`
5. Builds p(damage) and p(swstr) model scores from the updated pitch data
6. Stitches all season chunks into the final output files in `data/output/`
7. Merges model scores into the stitched output files
8. Runs Bayesian regression on the stitched hitter file
9. Updates `last_pull_date.txt` to today - 1

---

## Standard Daily Run

```bash
python run_daily_refresh.py
```

Default behavior:
- `--season 2026` — current season to pull and re-aggregate
- `--min-season 2015` — earliest season included in stitched output
- `--game-types R S F D L W` — all non-exhibition game types
- Reads start date from `data/logs/last_pull_date.txt` + 1 day
- Pulls through yesterday (today - 1)

### Dry run (print commands, no execution)

```bash
python run_daily_refresh.py --dry-run
```

---

## Manual Step-by-Step (if orchestrator fails mid-run)

If `run_daily_refresh.py` fails partway through, you can re-run individual steps. Each script is idempotent — re-running overwrites its outputs safely.

### Step 1 — Pull incremental data

```bash
python data_pull.py \
  --min-season 2026 --max-season 2026 \
  --start-date YYYY-MM-DD --end-date YYYY-MM-DD \
  --game-types R S F D L W \
  --out-file data/raw/pitch_data_2026_incremental.parquet
```

- `--start-date` = last pull date + 1 day (read from `data/logs/last_pull_date.txt`)
- `--end-date` = today - 1

### Step 2 — Accumulate into season parquet

This is done inline by `run_daily_refresh.py` (the `accumulate_pitch_data()` function). To run manually:

```python
# In a Python session:
from run_daily_refresh import accumulate_pitch_data
from pathlib import Path
accumulate_pitch_data(
    Path("data/raw/pitch_data_2026_incremental.parquet"),
    Path("data/raw/pitch_data_2026.parquet"),
)
```

### Step 3 — Re-aggregate current season chunk

```bash
python data_aggregate.py \
  --parquet-path data/raw/pitch_data_2026.parquet \
  --min-season 2026 --max-season 2026 \
  --chunk-by-season \
  --chunk-dir data/output/_season_chunks \
  --out-dir data/output
```

### Step 4a — Build p(damage) source tables

```bash
python build_p_damage_sources.py --parquet-path data/raw/pitch_data_2026.parquet
```

### Step 4b — Build p(swstr) source tables

```bash
python build_p_swstr_sources.py --parquet-path data/raw/pitch_data_2026.parquet
```

### Step 5 — Stitch all season chunks into final output files

```python
# In a Python session:
from run_daily_refresh import stitch_season_chunks
stitch_season_chunks(min_season=2015, current_season=2026)
```

### Step 6a — Merge p(damage) into aggregated files

```bash
python merge_p_damage_into_sources.py
```

### Step 6b — Merge p(swstr) into aggregated files

```bash
python merge_p_swstr_into_sources.py
```

### Step 7 — Apply Bayesian regression

```bash
python apply_regression_from_agg.py --hitters data/output/damage_pos_2015_2026.parquet
```

### Step 8 — Update last pull date

Update `data/logs/last_pull_date.txt` to the end date used in Step 1. On Windows, use Python to write it — `echo >` produces UTF-16 which the pipeline cannot read:

```bash
python -c "open('data/logs/last_pull_date.txt', 'w', encoding='utf-8').write('YYYY-MM-DD')"
```

---

## Key Files and Directories

| Path | Purpose |
|------|---------|
| `data/logs/last_pull_date.txt` | Tracks last successfully pulled date; pipeline starts from this + 1 day |
| `data/raw/pitch_data_{season}.parquet` | Accumulated pitch-level data for the season; deduped on `(pa_id, pitch_of_ab)` |
| `data/raw/pitch_data_{season}_incremental.parquet` | Temporary incremental pull; safe to delete after accumulation |
| `data/output/_season_chunks/` | Per-season aggregate chunks for all output tables |
| `data/output/damage_pos_2015_{season}.parquet` | Stitched hitter aggregate (main input to regression) |
| `data/output/pitcher_stuff_new.parquet` | Stitched pitcher aggregate |
| `data/output/hitters_regressed.parquet` | Bayesian-regressed hitter stats |
| `data/output/pitchers_regressed.parquet` | Bayesian-regressed pitcher stats |
| `data/output/pitch_types_regressed.parquet` | Bayesian-regressed pitch type stats |
| `config/stability_constants.csv` | Regression priors (mu, K) by season/level/stat — only updated end-of-season |
| `config/stability_config.yml` | Which stats to regress and by what method (rate vs mean) |

---

## Season Chunk Naming Convention

Season chunks are stored in `data/output/_season_chunks/` with this naming pattern:

- **Hitter aggregate**: `damage_pos_{min}_{max}.season_{YYYY}.parquet`
  - e.g. `damage_pos_2026_2026.season_2026.parquet`
- **All other tables**: `{stem}.season_{YYYY}.parquet`
  - e.g. `hitter_pctiles.season_2026.parquet`, `pitcher_stuff_new.season_2026.parquet`

**Important**: Only one chunk per season per table should exist. If you see multiple chunks for the same season (e.g. `damage_pos_2015_2025.season_2015.parquet` AND `damage_pos_2015_2015.season_2015.parquet`), the older multi-season-prefix files are stale and must be deleted before running the stitch step. Use:

```bash
# Example: remove stale 2015-2025 multi-season chunks
rm data/output/_season_chunks/damage_pos_2015_2025.season_*.parquet
```

---

## Game Type Reference

| Code | Group | Notes |
|------|-------|-------|
| `R` | Regular Season | Standard season games |
| `S` | Spring Training | Cactus/Grapefruit League |
| `F` | Postseason | Wild Card game |
| `D` | Postseason | Division Series |
| `L` | Postseason | Championship Series |
| `W` | Postseason | World Series |
| `A`, `C`, `E` | — | Exhibition/All-Star; excluded from all pulls |

The default game type group shown in the app is **Regular Season**. The `game_type_group` column is `null` for all data aggregated before the 2026 season refactor — these are treated as Regular Season throughout the app.

---

## Starting a New Season

When a new season begins (e.g. transitioning from 2026 to 2027):

1. Update `--season` default in `run_daily_refresh.py` from `2026` → `2027`
2. Set `data/logs/last_pull_date.txt` to the day before the new season's first game
3. No other changes are needed — the pipeline creates new `pitch_data_2027.parquet` and `damage_pos_2015_2027.parquet` automatically

---

## End-of-Season Tasks (done once per year)

These are NOT part of the daily refresh:

- **Recompute stability constants**: Run `stability_constants.py` on the full completed season's data. Output goes to `config/stability_constants.csv`. This requires a separate process and Rob's approval before overwriting.
- **Archive season data**: The current season's raw parquet (`pitch_data_{year}.parquet`) can be kept as-is; it serves as the historical record.

---

## Troubleshooting

### Pipeline fails at `build_p_damage_sources.py` or `build_p_swstr_sources.py`

**Symptom**: `KeyError: 'game_type_group'`
**Cause**: `score_dataframe()` filters to required columns before groupby; `game_type_group` must be in the required list.
**Fix**: Verify `"game_type_group"` is in the `required` list inside `score_dataframe()` in both build scripts.

### Regression produces mostly null `*_reg` values

**Symptom**: `hitters_regressed.parquet` has <20% non-null values for reg stats.
**Cause**: Polars join does not match `NULL = NULL` by default. If `game_type_group` is null for historical data, the `merge_frames` join in `apply_regression_from_agg.py` silently drops those rows.
**Fix**: Ensure `merge_frames` uses `nulls_equal=True` in its join call (line ~226 of `apply_regression_from_agg.py`).

### Stitched `damage_pos` has 2× as many rows as expected

**Symptom**: `damage_pos_2015_{year}.parquet` has ~2× the expected row count; `_season_chunks/` contains both `damage_pos_2015_2025.season_YYYY.parquet` AND `damage_pos_YYYY_YYYY.season_YYYY.parquet` files.
**Cause**: Stale multi-season chunks from a prior full-season aggregate run were not removed.
**Fix**: Delete the stale `damage_pos_2015_2025.season_*.parquet` files from `_season_chunks/`, then re-run the stitch step.

### App AR/Comps tabs show all "None" values

**Symptom**: Hitter/Pitcher AR tabs display blank stat cells; Comps tabs show no results or all-None rows.
**Cause**: `hitters_regressed.parquet` has null reg values (see regression issue above), or `damage_pos` has duplicate rows causing incorrect merge behavior.
**Fix**: Resolve the regression null issue above, then restart the Streamlit app to clear cache.

### `scipy` import errors when loading GAM models

**Symptom**: `ModuleNotFoundError: No module named 'scipy.optimize._trustregion_constr.projections'`
**Fix**: `pip install --force-reinstall scipy` inside the venv.

---

## Deployment

After a successful pipeline run, push the updated data files and code to the remote repo (with Rob's approval):

```bash
git add data/output/ data/logs/ data/raw/pitch_data_*.parquet
git commit -m "Daily refresh YYYY-MM-DD"
git push
```

The Streamlit app at py-players.streamlit.app auto-deploys on push to `main`.

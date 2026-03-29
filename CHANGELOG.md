# Changelog

---

## 2026-03-28 — Repo script reorganization

### Structure
**`pipeline/`** *(new folder, tracked)*
- Moved 10 active pipeline scripts here: `data_pull.py`, `data_aggregate.py`, `build_p_damage_sources.py`, `build_p_swstr_sources.py`, `build_p_swstr_hitter_sources.py`, `build_p_damage_hitter_sources.py`, `merge_p_damage_into_sources.py`, `merge_p_swstr_into_sources.py`, `merge_model_outputs_into_damage_pos.py`, `apply_regression_from_agg.py`.

**`model_dev/`** *(new folder, gitignored)*
- Moved 4 non-pipeline model scripts here: `apply_regression.py`, `backfill_model_outputs.py`, `catboost_predictor.py`, `stability_constants.py`.

**`scripts/`** *(new folder, gitignored)*
- Moved `convert_csv_to_parquet.py` here.

**`run_daily_refresh.py`**
- Updated all 10 subprocess path references from `HERE / "script.py"` to `HERE / "pipeline" / "script.py"`.

**`.gitignore`**
- Added `model_dev/` and `scripts/` entries.

---

## 2026-03-27 — Column display cleanup + HR count added to all pages

### App pages
**`app/pages/hitters.py`**
- Removed `p_Swing_pct`, `p_SwStr_pct`, `p_SwStr_with_loc_pct`, `p_Damage_pct`, `p_Damage_with_loc_pct` from Individual Stats and AR columns; renamed `p_Swing_with_loc_pct` → `"pSwing (%)"` to replace the removed `p_Swing_pct` label.
- Added `HR` (count) to all four functions (Individual Stats, Percentiles, AR, Splits); green=higher coloring (default).

**`app/pages/pitchers.py`**
- Removed `p_SwStr_with_loc_pct`, `p_Swing_with_loc_pct`, `p_Damage_with_loc_pct` (and `_reg` variants) from Individual Stats and AR columns; removed `"pDamage+Loc/BBE (%)"` from all `reverse_cols` sets.
- Added `HR` (count) to all four functions (Individual Stats, Percentiles, AR, Splits); red=higher coloring via `reverse_cols`.

**`app/pages/pitches.py`**
- Removed `p_SwStr_with_loc_pct`, `p_Swing_with_loc_pct`, `p_Damage_with_loc_pct` (and `_reg` variants) from Shapes/AR columns; removed stale `pDamage+Loc/BBE (%)` and `pDamage/BBE (%)` from `reverse_cols` where not present in columns.
- Added `rel_z`, `rel_x`, `ext` (Vertical/Horizontal Release, Extension) to Shapes, AR, and Splits columns.
- Added `HR` (count) to all four functions (Shapes, Percentiles, AR, Splits); red=higher coloring via `reverse_cols`.

### Data pipeline
**`data_aggregate.py`**
- Added `(pl.col("pitch_outcome") == "HR").sum().alias("HR")` to `build_hitters()`, `build_pitchers()`, and `build_pitch_types()` aggregation blocks.
- All season chunk parquets (2015–2026) rebuilt to include HR counts.

### Config
**`app/config.py`**
- Added `"HR"` to `DEFAULT_NO_FORMAT_COLS` to display as integer (no decimal).

---

## 2026-03-26 — p(swing), p(swstr), p(damage) backfill + hitter model pipeline

### Build scripts
**`build_p_swstr_sources.py`**
- Extended `aggregate_outputs`, `collapse_duplicates`, and `score_dataframe` to produce `Swing_pct`, `p_Swing_pct`, `p_Swing_with_loc_pct` alongside existing SwStr outputs.

**`build_p_swstr_hitter_sources.py`** *(new)*
- Batter-level variant of `build_p_swstr_sources.py`. Groups the same swing/whiff model predictions by `batter_mlbid`/`hitter_name`. Outputs `data/raw/hitter_p_swstr.parquet`.

**`build_p_damage_hitter_sources.py`** *(new)*
- Batter-level variant of `build_p_damage_sources.py`. Groups damage model predictions by batter. Outputs `data/raw/hitter_p_damage.parquet`.

### Merge scripts
**`merge_p_swstr_into_sources.py`**
- Added `Swing_pct`, `p_Swing_pct`, `p_Swing_with_loc_pct` to both pitcher and pitch-type merge. Added `p_Swing_n` alias column for regression n-resolution.

**`merge_model_outputs_into_damage_pos.py`** *(new)*
- Merges `hitter_p_swstr.parquet` and `hitter_p_damage.parquet` into the latest `damage_pos_2015_YYYY.parquet`. Adds `Swing_pct`, `p_SwStr_pct`, `p_Swing_pct`, `p_Damage_pct` (+ loc variants and n columns).

### Regression
**`apply_regression_from_agg.py`**
- Added `Swing_pct`, `p_Swing_pct`, `p_Swing_with_loc_pct` to `PERCENT_MEAN_STATS`.

**`config/stability_config.yml`**
- Pitchers/pitch-types: added `p_Swing_pct` and `p_Swing_with_loc_pct` entries.
- Hitters: added `Swing_pct`, `p_Swing_pct`, `p_Swing_with_loc_pct`, `p_SwStr_pct`, `p_SwStr_with_loc_pct`, `p_Damage_pct`, `p_Damage_with_loc_pct` as mean stats.

**`config/stability_config_modeled_pitching.yml`**
- Added `p_Swing_pct` and `p_Swing_with_loc_pct` for both pitchers and pitch_types.

### Backfill
**`backfill_model_outputs.py`** *(updated)*
- Added full hitter pipeline (steps 6-8): batter source builds, combine with current season, merge into damage_pos, hitter regression.
- Added `--skip-pitchers` / `--skip-hitters` flags for partial re-runs.

### App pages
**`app/pages/pitchers.py`**
- Non-AR tab: added `Swing_pct`, `p_Swing_pct`, `p_Swing_with_loc_pct` columns.
- AR tab: added `p_SwStr_pct_reg`, `p_SwStr_with_loc_pct_reg`, `p_Swing_pct_reg`, `p_Swing_with_loc_pct_reg`, `p_Damage_pct_reg`, `p_Damage_with_loc_pct_reg`.

**`app/pages/hitters.py`**
- Non-AR tab: added `Swing_pct`, `p_Swing_pct`, `p_Swing_with_loc_pct`, `p_SwStr_pct`, `p_SwStr_with_loc_pct`, `p_Damage_pct`, `p_Damage_with_loc_pct`.
- AR tab: added corresponding `_reg` columns.

---

## 2026-03-26

### Streamlit App Modularization (`refactor/modular-app` branch)

**`damage_streamlit.py`**
- Reduced from ~5,935 lines to 239 lines. Now serves as routing-only entry point: page config, CSS, auth/subscription gate, session timeout, and `st.navigation` registry.
- All page functions, helpers, constants, and data loads moved to `app/` modules (see below).

**`app/config.py`** *(new)*
- All module-level constants extracted from `damage_streamlit.py`: `DATA_DIR`, display sets, Stripe links, level/position maps, hitter and pitcher feature/comp config constants.

**`app/auth.py`** *(new)*
- Straight lift of all auth/subscription functions: `_get_stripe_api_key`, `_infer_return_url`, `_get_user_email`, `_get_subscription_exempt_emails`, `_is_subscription_exempt_user`, `_create_billing_portal_url`, `_resolve_subscription_status`, `_is_user_subscribed`. No logic changes.

**`app/data_loader.py`** *(new)*
- `_load_csv_cached`, `_optimize_dataframe_memory`, `load_csv`, `load_damage_df`. All `st.cache_data` replaced with `st.cache_resource`.

**`app/datasets.py`** *(new)*
- Module-level data loads, column normalization, pitch_group backfill, regressed merges, and MLB equivalency table construction.

**`app/filters.py`** *(new)*
- All filter/UI helpers: `season_options`, `filter_by_values`, `team_options`, `filter_by_team_token`, `position_options`, `filter_by_positions`, `player_id_options`, `numeric_filter`, `game_type_group_options`, `filter_by_game_type_group`, `pitcher_workload_filter`, `download_button`, `apply_column_filters`.

**`app/utils.py`** *(new)*
- Column normalization helpers, display maps, similarity label helpers, regressed merge, and MLB equivalency computation (`_build_hitter_mlb_equivalencies`, `_build_pitcher_mlb_equivalencies`).

**`app/viz.py`** *(new)*
- `render_table` and all supporting plot/visualization helpers.

**`app/pages/home.py`** *(new)*
- `home_page`.

**`app/pages/hitters.py`** *(new)*
- `hitter_individual_stats`, `hitter_percentiles`, `hitter_comps`, `hitter_mlb_equivalencies`, `hitter_ar`, `hitter_splits`.

**`app/pages/pitchers.py`** *(new)*
- `pitcher_individual_stats`, `pitcher_percentiles`, `pitcher_comps`, `pitcher_mlb_equivalencies`, `pitcher_ar`, `pitcher_splits`.

**`app/pages/pitches.py`** *(new)*
- `pitch_shapes_outcomes`, `pitch_ar`, `pitch_percentiles`, `pitch_comps`, `pitch_splits`.

**`app/pages/teams.py`** *(new)*
- `team_hitting`, `team_pitching`.

**`app/pages/league.py`** *(new)*
- `league_hitting`, `league_pitching`, `league_pitch_level`.

**`app/pages/parks.py`** *(new)*
- `park_data_page`.

**`app/pages/glossary.py`** *(new)*
- `glossary_hitting`, `glossary_pitching`.

**Global**
- All `st.cache_data` occurrences replaced with `st.cache_resource` throughout `app/` modules.
- All 26 page functions and 16 modules pass syntax checks and import smoke tests.

---

## 2026-03-17

### Daily Pipeline Infrastructure

**`data_pull.py`**
- Added `--game-types` CLI arg (default: `['R']`). Replaced hard-coded `AND a.game_type = 'R'` SQL filter with a parameterized `IN (...)` clause. Valid values: `R`=Regular Season, `S`=Spring Training, `F/D/L/W`=Postseason.
- Added `--start-date` / `--end-date` CLI args for incremental date-range pulls. SQL gains `AND a.game_date >= '...' AND a.game_date <= '...'` when provided.
- Same changes applied to `fetch_level_ids()` so level auto-detection respects the game type filter.

**`data_aggregate.py`**
- Added `GAME_TYPE_GROUP_MAP` constant and `_add_game_type_group()` helper. Derivation: `R` → `"Regular Season"`, `S` → `"Spring Training"`, `F/D/L/W` → `"Postseason"`, null/other → `"Regular Season"`.
- `_build_outputs()` now calls `_add_game_type_group(pitch)` after normalizations. `game_type_group` is propagated as a grouping key into all `build_*` functions, league/team aggregations, stuff grade intermediates, and `add_percentiles` calls.

**`build_p_damage_sources.py`**
- Replaced two-file external level-split pattern (`--level1-parquet` / `--no-level1-parquet`) with a single `--parquet-path` pointing to the local season accumulator.
- Added `game_type_group` to `GROUP_PITCHER` and `GROUP_PITCH_TYPE`. Added `_add_game_type_group()` helper.

**`build_p_swstr_sources.py`**
- Same changes as `build_p_damage_sources.py` — single `--parquet-path`, `game_type_group` added to group keys.

**`merge_p_damage_into_sources.py`**
- Added `game_type_group` to join keys (inferred from source parquet columns).
- Added `Damage_pct` (real damage/BBE%) to `value_cols` in weighted merge so it flows through alongside `p_Damage_pct` and `p_Damage_with_loc_pct`. Updated drop list accordingly.

**`merge_p_swstr_into_sources.py`**
- Added `game_type_group` to join keys.
- Added `SwStr_pct` (real swing-and-miss rate) to `value_cols` so it flows through alongside modeled values. Updated drop list accordingly.

**`apply_regression_from_agg.py`**
- Added `game_type_group` to key lists for all three datasets (hitters, pitchers, pitch_types). `_join_constants` ignores it naturally — no new constants required.
- `add_league_contact_baseline()` now groups by `game_type_group` when present, so league whiff baselines are computed per game type.

**`damage_streamlit.py`**
- Added `GAME_TYPE_GROUP_OPTIONS`, `GAME_TYPE_GROUP_NOTE`, `game_type_group_options()`, and `filter_by_game_type_group()` helper utilities.
- Added "Game Type" selectbox (Regular Season / Spring Training / Postseason, default Regular Season) to the filter column of all 17 data pages.
- Added contextual `st.info()` note on each page when Spring Training or Postseason is selected.
- Old data rows lacking `game_type_group` (2015–2025 historical) are treated as Regular Season.

**`run_daily_refresh.py`** *(new)*
- Orchestrator script for the full incremental daily pipeline. Reads `data/logs/last_pull_date.txt`, pulls incremental rows, accumulates them into `data/raw/pitch_data_{season}.parquet` with deduplication on `(pa_id, pitch_of_ab)`, re-aggregates only the current season chunk, stitches all season chunks into final output files, runs build/merge probability steps, and applies regression. Updates `last_pull_date.txt` on success.
- Supports `--dry-run` to preview all commands without executing.

**`data/logs/last_pull_date.txt`** *(new)*
- Initialized to `2025-09-28` (last date of 2025 regular season data). First run will pull 2026 season data from opening day onward.

### Pipeline Bug Fixes (first-run validation)

**`build_p_damage_sources.py`**
- Fixed `KeyError: 'game_type_group'` in `score_dataframe()`: added `"game_type_group"` to the `required` columns list so it survives the `available` filter before `aggregate_outputs()`.

**`build_p_swstr_sources.py`**
- Same fix as `build_p_damage_sources.py` — `"game_type_group"` added to `required` in `score_dataframe()`.

**`apply_regression_from_agg.py`**
- Fixed `ValueError: Frame missing join keys ['game_type_group']` in `merge_frames()`: compute `effective_keys = [c for c in keys if c in df.columns]` and use it for `base`, frame generation, and `merge_frames` call so historical files without `game_type_group` are handled gracefully.
- Fixed `InvalidOperationError: division with 'String' datatypes`: added `cast(pl.Float64, strict=False)` on `raw` and `n` columns in `apply_mean_from_agg()` to handle cases where a source column is String-typed (e.g., `arm_angle` in 2026 spring training data is all-null String).

**`run_daily_refresh.py`**
- Fixed Unicode encode error on Windows: replaced `→` and `–` characters in `print()` calls with ASCII equivalents (`->` and `-`).
- Fixed hitters regression using stale `damage_pos_2015_2025.parquet`: now passes `--hitters data/output/damage_pos_{min_season}_{season}.parquet` explicitly to `apply_regression_from_agg.py` so the stitched multi-season file is always used.

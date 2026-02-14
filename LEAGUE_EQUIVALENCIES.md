# League Equivalencies (Hitters)

This document describes how hitter MLB equivalencies are currently built in `damage_streamlit.py`.

## Scope

- Dataset scope: hitter season-level rows from:
  - `damage_pos_2015_2025.parquet` (base stats + PA)
  - `hitters_regressed.parquet` (`*_reg` metrics)
- Level mapping:
  - `1 = MLB`
  - `11 = Triple-A`
  - `14 = Low-A`
  - `16 = Low Minors`

## Output

For each hitter metric `X_reg`, the pipeline produces:

- `X_reg_mlb_eq`: MLB-equivalent version of that metric

These translated columns are used by:

- `Hitter MLB Equivalencies` page
- `Hitter Comps` page when `Use MLB-equivalent translated stats` is enabled

## Metrics Included

All numeric columns ending in `_reg` are eligible (except `reg_prop`).

## Training Data Preparation

1. Build player-season-level rows:
   - Group by `batter_mlbid, season, level_id`
   - `PA` is summed
   - metrics are averaged
2. Compute per-`season x level_id` moments for each metric:
   - mean and std (population std, `ddof=0`)
3. Convert metrics to z-scores within `season x level_id`:
   - `z = (x - mean_level_season) / std_level_season`

## Pair Construction (Training Sample)

Regressions are fit from both:

1. Same-season transitions:
   - source and destination are in the same season
2. Adjacent-season transitions:
   - source in season `n`, destination in season `n+1`

For both pair types, only promotion-direction edges are used:

- `16 -> 14`
- `14 -> 11`
- `11 -> 1`

## Edge Thresholds

Minimum PA for pair inclusion:

- `11 -> 1`: `src_PA >= 50` and `dst_PA >= 50`
- `14 -> 11`: `src_PA >= 50` and `dst_PA >= 50`
- `16 -> 14`: `src_PA >= 10` and `dst_PA >= 10`

## Edge Regression Fit

For each metric and edge, fit weighted linear model in z-space:

- `z_dst = a + b * z_src`
- weights: `sqrt(src_PA * dst_PA)`

Then apply shrinkage toward prior `(a=0, b=0.5)`:

- reliability: `n / (n + 50)`
- `a_fit = reliability * a_raw + (1 - reliability) * 0`
- `b_fit = reliability * b_raw + (1 - reliability) * 0.5`

Clipping:

- `a_fit` clipped to `[-1.5, 1.5]`
- `b_fit` clipped to `[-0.25, 1.25]`

## Chaining to MLB

Direct MLB edge:

- `11 -> 1` uses fitted `(a, b)`

Chained edges:

- `14 -> 1` = compose `(14->11)` with `(11->1)`
- `16 -> 1` = compose `(16->14)` with `(14->11)` and `(11->1)`

Linear composition:

- if `z_mid = a1 + b1*z_src`
- and `z_mlb = a2 + b2*z_mid`
- then `z_mlb = (a2 + b2*a1) + (b2*b1)*z_src`

## Translating Back to Raw MLB Scale

For each non-MLB row and metric:

1. Compute source z-score at its own `season x level`
2. Apply level-to-MLB `(a, b)` to get `z_mlb_pred`
3. Convert to raw MLB scale using same-season MLB mean/std:
   - `x_mlb_eq = mlb_mean_season + z_mlb_pred * mlb_std_season`

For MLB rows (`level_id=1`):

- `x_mlb_eq = x_raw` (pass-through)

## Post-fit Directional Calibration

After model prediction, non-MLB rows are calibrated:

1. No-improvement guard:
   - For higher-is-worse metrics: `pred = max(pred, raw)`
   - For all others: `pred = min(pred, raw)`
2. Minimum directional shift:
   - Direction map specifies expected MLB movement (`up` or `down`) by metric
   - Shift magnitude is based on absolute source-vs-MLB seasonal mean gap:
     - base scale: `0.75`
     - overrides:
       - `LA_gte_20_reg`: `1.25`
       - `LA_lte_0_reg`: `1.5`
     - floors:
       - `LA_gte_20_reg`: `2.0`
       - `LA_lte_0_reg`: `2.0`

## Coefficients Table in App

The app exposes coefficient rows with:

- `src_level`, `dst_level`, `metric`
- fitted `a`, `b`
- sample count `n`
- threshold metadata (`min_src_pa`, `min_dst_pa`)
- `fit_type`:
  - `intra+inter-season` for directly fit edges
  - `chained` for composed edges

## Notes / Caveats

- `14->11` and especially `16->14` can remain sample-limited after thresholds.
- Directional calibration intentionally biases outputs toward expected translation direction; it is not purely unconstrained regression output.
- All transformations are season-contextual (season-specific level means/stds).

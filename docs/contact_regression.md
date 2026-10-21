# Direct Contact Over Expected regression

Observed COE remains expected whiff percentage minus actual whiff percentage.
AR COE estimates the underlying season-level mean on that same scale:

`AR COE = (observed COE * swings + league COE * K) / (swings + K)`

The target is season-level COE in percentage points. Inputs are observed COE,
recorded swings, season and level. This is analytic empirical-Bayes mean shrinkage;
feature importance rankings do not apply. Configuration is in
`config/contact_regression.yml`.

For calibration, each regular-season swing contributes
`100 * (pred_whiff_loc - whiff)`. Finite predictions and binary whiff outcomes are
required. Player-season-level means and within-player variances are calculated;
players with at least 20 valid swings contribute to stability estimation, with at
least 10 players required per context. The existing mean-stat method estimates
between-player variance after removing sampling variance, then uses
`K = within-swing variance / between-player variance`.

League COE is weighted by all valid swings, including small samples. Historical
completed seasons have their own fitted K. The latest season uses the previous
season's fitted K at the same level, with a current-season fit when no prior exists.
The daily `blend_constants.py` step keeps this convention and refreshes league COE.
The 2026 MLB prior is +3.4703 COE with K=36.4201 swings, fitted on 2025.

This estimates the underlying mean, rather than projecting additional plate
appearances or the hitter's future pitch mix. Larger samples retain more of their
observed COE. It adjusts neither whiffs separately nor observed COE's reference point.

## Validation

A chronological held-out check used 2026 MLB regular-season swings before July 1
to predict COE from July 1 onward, with at least 100 valid swings in each period.
The 365 eligible hitters used K fitted on 2025 and a league mean calculated only
from the first period. The held-out period was not used in those parameters.

| Predictor | Held-out R-squared | RMSE, percentage points |
|---|---:|---:|
| Observed first-period COE | 0.6622 | 3.5876 |
| Direct AR first-period COE | 0.6852 | 3.4632 |

The held-out target is another sample of season-level skill, not an observable
noise-free true-talent value. This check supports modest shrinkage; it does not
establish a fully optimized forecasting model.

Five focused tests cover signed shrinkage, sample size, preservation of observed
COE, swing filtering, weighted league means and explicit missing-prior behavior.
All available estimates must lie between observed COE and the prior mean.

There are 529 non-null observed rows without enough regular-season calibration
data: 2021 level 16 (123), 2024 level 16 (104), 2026 level 586 (26), and 2026 level
51 postseason (276). Their AR COE is null, and regeneration emits a warning.
Observed COE remains available. No MLB regular-season context is missing a prior.

## Rebuilding

Run `pipeline/contact_stability.py --parquet-path` with the disjoint yearly
`data/raw/_hist_seasons/pitch_data_*.parquet` files plus
`data/raw/pitch_data_2026.parquet`, then run
`pipeline/apply_regression_from_agg.py --hitters data/output/damage_pos_2015_2026.parquet`.
The builder preserves constants for other metrics and seasons not supplied.
Daily refresh already calls the updated blend and regression steps.

Original constants and hitter regression output are backed up in
`data/backups/direct_coe_20261021_004025`. Restore those files together with the
prior regression/config source to roll back. No repository push is part of this change.

# Comps metric coverage â€” October 23, 2026

Every non-count metric on the Hitter, Pitcher and Individual Pitches main stats pages is selectable for similarity. Counts and count-derived substitute rates are excluded; pitch usage is allowed. Existing presets remain editable.

Hitter and Pitcher skill metrics use their auto-regressed / MLB-equivalent versions. Bat tracking (including Miss Distance) and pitch usage use observed values in both modes, without adding fake translated versions. Existing explicitly untranslated grade/trait sources retain their prior treatment. Individual Pitches remains observed, with only fastball VAA converted by the existing toggle.

The selector sources are listed in `app/comparison_metrics.py`. Missing modeled values never fall back to observed values; the existing comparison mean-imputation behavior is unchanged. The selected-season and result tables use the modeled swing/whiff columns rather than showing an observed duplicate first.

## Additional regression outputs

The existing season/level binomial empirical-Bayes estimator is reused for Whiff% (hitters and pitchers). The audit also found missing hitter SwStr%, pitcher Swing%, and pitcher Zone% outputs; these now receive the same treatment so their Comps options are modeled. Individual Pitches is unaffected.

Targets: the observed percentages named below. Inputs/features: event numerator, opportunity denominator, season, and level. Method: estimate mean and noise-adjusted between-player variance, convert to prior strength K, and shrink the observed rate toward that context mean. This is not a learned feature-weight model: the observation receives weight n/(n+K), and the context mean receives K/(n+K). The method gives low-sample rows more help from the league average.

Sampling: fit regular-season player-seasons with at least 20 opportunities and at least 10 players per season/level. Unsupported priors return missing modeled values. Counts are pooled by player/season/level before fitting. Rate definitions, feature description and sampling live in `config/comparison_rates.yml`.

Evaluation: predict the following regular season's observed rate for the same player and level, requiring at least 20 target-season opportunities. Prior estimates and predictions use only the source season. Results measure year-to-year predictive agreement, not prediction of an unobserved true skill. RMSE is in percentage points.

| Population | Target | Pairs | RÂ² | RMSE |
|---|---|---:|---:|---:|
| hitters | whiff_pct | 8,333 | 0.488 | 5.49 |
| hitters | SwStr | 8,749 | 0.412 | 3.34 |
| pitchers | whiff_pct | 9,270 | 0.248 | 5.60 |
| pitchers | Swing_pct | 9,620 | 0.155 | 4.27 |
| pitchers | Zone | 9,619 | 0.182 | 4.07 |

The new rates are registered in `config/stability_config.yml`, alongside Whiff vs. Secondaries and Whiff vs. 95+. Historical priors were initialized with the same rate estimator and stored in `config/stability_constants.csv` and `.parquet`. The 2026 rows were produced by the existing `build_blended_rows` function: K is retained from 2025; mu uses prior-only, player-count-weighted blending, or current-only according to the existing 10-player and 80% thresholds. New definitions restrict daily fitting to regular-season pitches.

`pipeline/comparison_rates.py` only aliases retained counts during refresh. It does not fit priors in that path. The normal regression loop calls the existing `apply_rate_from_agg` with stored constants. A `require_prior` flag preserves the requested missing-value behavior for new rates; existing rates retain their previous behavior. The separate experimental comparison-priors artifact was archived, and the established stability tables are now authoritative.

Only new prior rows were appended; all 5,365 pre-existing constant rows (including local edits) were preserved. Backfilled hitter/pitcher new columns use the standard daily regression output. Original data columns remain unchanged. Before-alignment snapshots are under `data/backups/comparison_alignment/20261023`; initial backups remain under `data/backups/comparison_metrics/20261023`.

The existing MLB-equivalency builders pick up the new `_reg` columns automatically; hitter Whiff% and SwStr% follow the higher-is-worse direction, and pitcher Whiff% follows the higher-is-better direction. Translated rate outputs are bounded to 0â€“100. No new pitch-level translation was added.

Validation: 28 focused tests passed. All six real-data Comps views (both toggle
settings for each role) rendered their selected-season and results tables with
all main-page skill metrics selected. The data import completed without Python
warnings. A full regression-step run to a separate output directory reproduced
the new columns, and all original columns in the two backfilled production files
were verified unchanged. That run also reported the existing direct-COE warning:
529 rows lack priors (2026 levels 51/586 and 2021/2024 level 16); this change does
not create or repair those COE priors. Existing local stability-constant edits
were preserved when adding the new rate priors.

The daily-refresh dry run also passed and includes the regression step with the
current multi-season hitter input. Backfilled new columns match the full daily
regression output exactly at its stored precision.

Alignment checks confirm all 20 supported 2026 contexts retain their matching 2025 K. Tests compare new and existing whiff measures under all three seasonal-blending branches, verify stored-prior scoring and missing-prior behavior, and exercise the daily blend against mixed regular-season and spring-training pitches.

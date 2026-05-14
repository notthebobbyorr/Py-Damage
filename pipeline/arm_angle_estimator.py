"""Inference helper for the trained `arm_angle_right` estimator.

The model is a sklearn `Pipeline` (PolynomialFeatures -> StandardScaler -> Ridge)
trained by `model_dev/train_arm_angle.py` on rows where Statcast publishes
arm_angle_right (2020 + 2024 sparse + 2025). It takes four handedness-agnostic
features and produces an estimated arm_angle_right in degrees.

The aggregated pitcher / pitch_types tables already hold the per-pitcher means
of release_x, release_z, ext that the model needs. `pitcher_height` is added by
the aggregation step (see `build_pitchers` / `build_pitch_types`).

Public functions:
    fill_arm_angle_right(df, *, rel_x="rel_x", rel_z="rel_z", ext="ext", height="pitcher_height")
        Returns the input df with a new column `inf_arm_angle` containing
        the model's prediction (or null when any required feature is missing or
        the model isn't loaded).

    coalesce_arm_angle_right(df, observed="arm_angle_right", est="inf_arm_angle")
        Replaces nulls in `observed` with values from `est`, drops `est`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

_REPO = Path(__file__).resolve().parent.parent
MODEL_PATH = _REPO / "models" / "arm_angle_est.joblib"

_payload = None
_load_error: Exception | None = None


def _load() -> dict | None:
    """Lazy-load the model bundle. Returns None if the model file is missing
    (e.g. during a fresh checkout before train_arm_angle.py has run)."""
    global _payload, _load_error
    if _payload is not None:
        return _payload
    if _load_error is not None:
        return None
    try:
        import joblib

        _payload = joblib.load(MODEL_PATH)
        return _payload
    except FileNotFoundError as exc:
        _load_error = exc
        print(
            f"WARNING: arm_angle_right estimator not found at {MODEL_PATH}. "
            f"Run `py -3.13 model_dev/train_arm_angle.py` to generate it. "
            f"Predictions will be null."
        )
        return None
    except Exception as exc:  # pragma: no cover
        _load_error = exc
        print(
            f"WARNING: failed to load arm_angle_right estimator from {MODEL_PATH}: "
            f"{exc}. Predictions will be null."
        )
        return None


def fill_arm_angle_right(
    df: pl.DataFrame,
    *,
    rel_x: str = "rel_x",
    rel_z: str = "rel_z",
    ext: str = "ext",
    height: str = "pitcher_height",
) -> pl.DataFrame:
    """Add `inf_arm_angle` column. Null when any feature is missing."""
    if df.is_empty():
        return df.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("inf_arm_angle")
        )
    payload = _load()
    if payload is None:
        return df.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("inf_arm_angle")
        )

    model = payload["model"]
    feat_cols = payload["feature_cols"]  # ["abs_rel_x", "rel_z", "ext", "pitcher_height"]

    # Required source columns must exist; if any are missing, skip imputation.
    needed = {rel_x, rel_z, ext, height}
    missing = needed - set(df.columns)
    if missing:
        print(
            f"WARNING: arm_angle estimator missing input columns {sorted(missing)}; "
            f"skipping imputation."
        )
        return df.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("inf_arm_angle")
        )

    feat_df = df.select(
        pl.col(rel_x).abs().alias("abs_rel_x"),
        pl.col(rel_z).alias("rel_z"),
        pl.col(ext).alias("ext"),
        pl.col(height).cast(pl.Float64, strict=False).alias("pitcher_height"),
    ).to_pandas()

    feat_df = feat_df[feat_cols]  # reorder to model's expected feature order
    valid = feat_df.notna().all(axis=1).to_numpy()
    preds = np.full(len(df), np.nan, dtype=float)
    if valid.any():
        preds[valid] = model.predict(feat_df.loc[valid].to_numpy())
    return df.with_columns(pl.Series("inf_arm_angle", preds, dtype=pl.Float64))


def coalesce_arm_angle_right(
    df: pl.DataFrame,
    observed: str = "arm_angle_right",
    est: str = "inf_arm_angle",
) -> pl.DataFrame:
    """Replace nulls in `observed` with `est`, then drop `est`."""
    if df.is_empty() or observed not in df.columns or est not in df.columns:
        return df.drop([c for c in (est,) if c in df.columns])
    return df.with_columns(
        pl.col(observed).fill_null(pl.col(est)).alias(observed),
    ).drop(est)

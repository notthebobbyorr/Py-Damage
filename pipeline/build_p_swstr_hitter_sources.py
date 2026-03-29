"""
Batter-level p(swing) and p(swstr) source builder.

Scores the same swing / whiff models as build_p_swstr_sources.py but
groups results by batter (batter_mlbid / hitter_name) instead of pitcher.
Output is a single parquet: hitter_p_swstr.parquet in data/raw/.

Columns produced per batter-season-level row:
  n                      – pitch count
  Swing_pct              – actual swing rate (%)
  p_Swing_pct            – model-expected swing rate, no location (%)
  p_Swing_with_loc_pct   – model-expected swing rate, with location (%)
  SwStr_pct              – actual swinging-strike rate (%)
  p_SwStr_pct            – model-expected SwStr rate, no location (%)
  p_SwStr_with_loc_pct   – model-expected SwStr rate, with location (%)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from build_p_swstr_sources import (
    WHIFF_NUM_FEATURES,
    WHIFF_CAT_FEATURES,
    WHIFF_BASE_NUM_FEATURES,
    _add_game_type_group,
    predict_in_batches,
    aggregate_outputs,
    collapse_duplicates,
    _upsert_parquet,
)

HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"
RAW_DIR = HERE / "data" / "raw"

GROUP_BATTER = ["batter_mlbid", "hitter_name", "season", "level_id", "game_type_group"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build batter-level p(swing) and p(swstr) source table."
    )
    parser.add_argument(
        "--parquet-path", type=Path,
        default=Path(r"C:\Users\orrro\Documents\Baseball_Env\pitch_data_2015_2025_all_levels.parquet"),
        help="Path to input pitch-level parquet.",
    )
    parser.add_argument(
        "--loc-model", type=Path,
        default=MODELS_DIR / "is_swing_catboost_model_with_locations.cbm",
    )
    parser.add_argument(
        "--base-model", type=Path,
        default=MODELS_DIR / "is_swing_catboost_model.cbm",
    )
    parser.add_argument(
        "--whiff-loc-model", type=Path,
        default=MODELS_DIR / "p_whiff_v2_with_locations.cbm",
    )
    parser.add_argument(
        "--whiff-base-model", type=Path,
        default=MODELS_DIR / "p_whiff_v2.cbm",
    )
    parser.add_argument(
        "--batter-output", type=Path,
        default=RAW_DIR / "hitter_p_swstr.parquet",
        help="Output path for batter-level source parquet.",
    )
    parser.add_argument("--batch-size", type=int, default=50_000)
    return parser.parse_args()


def score_dataframe(
    df: pd.DataFrame,
    swing_loc_model: CatBoostClassifier,
    swing_base_model: CatBoostClassifier,
    whiff_loc_model: CatBoostClassifier,
    whiff_base_model: CatBoostClassifier,
    batch_size: int,
) -> pd.DataFrame:
    """Score pitch data and aggregate to batter level. Returns one DataFrame."""
    required = sorted(set(
        WHIFF_NUM_FEATURES + WHIFF_CAT_FEATURES + [
            "whiff", "swing", "pitcher_role",
            "batter_mlbid", "hitter_name",
            "season", "level_id", "pitch_tag", "game_type_group",
        ]
    ))
    available = [c for c in required if c in df.columns]
    df = df[available].copy()
    df = df[df["pitcher_role"].isin(["SP", "RP"])].copy()
    df["whiff"] = pd.to_numeric(df["whiff"], errors="coerce").fillna(0).astype("int8")
    df["swing"] = pd.to_numeric(df["swing"], errors="coerce").fillna(0).astype("int8")
    df["hitter_name"] = df["hitter_name"].astype("string").fillna("UNK")
    df["pitch_tag"] = df["pitch_tag"].astype("string").fillna("UNK")
    for col in WHIFF_CAT_FEATURES:
        df[col] = df[col].astype("string").fillna("UNK")

    df["pred_swing_loc"] = predict_in_batches(
        swing_loc_model, df, WHIFF_NUM_FEATURES + WHIFF_CAT_FEATURES, batch_size
    )
    df["pred_swing_base"] = predict_in_batches(
        swing_base_model, df, WHIFF_BASE_NUM_FEATURES + WHIFF_CAT_FEATURES, batch_size
    )
    df["pred_whiff_loc_v2"] = predict_in_batches(
        whiff_loc_model, df, WHIFF_NUM_FEATURES + WHIFF_CAT_FEATURES, batch_size
    )
    df["pred_whiff_base_v2"] = predict_in_batches(
        whiff_base_model, df, WHIFF_BASE_NUM_FEATURES + WHIFF_CAT_FEATURES, batch_size
    )
    df["pred_swstr_loc"] = df["pred_swing_loc"] * df["pred_whiff_loc_v2"]
    df["pred_swstr_base"] = df["pred_swing_base"] * df["pred_whiff_base_v2"]

    return aggregate_outputs(df, GROUP_BATTER)


def main() -> None:
    args = parse_args()

    swing_loc_model = CatBoostClassifier()
    swing_loc_model.load_model(args.loc_model.as_posix())
    swing_base_model = CatBoostClassifier()
    swing_base_model.load_model(args.base_model.as_posix())
    whiff_loc_model = CatBoostClassifier()
    whiff_loc_model.load_model(args.whiff_loc_model.as_posix())
    whiff_base_model = CatBoostClassifier()
    whiff_base_model.load_model(args.whiff_base_model.as_posix())

    if not args.parquet_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {args.parquet_path}")

    print(f"Scoring {args.parquet_path} (batter-level) ...")
    df = pd.read_parquet(args.parquet_path)
    df = _add_game_type_group(df)

    batter_out = score_dataframe(
        df, swing_loc_model, swing_base_model, whiff_loc_model, whiff_base_model, args.batch_size
    )
    batter_out = collapse_duplicates(batter_out, GROUP_BATTER)

    args.batter_output.parent.mkdir(parents=True, exist_ok=True)
    batter_out = _upsert_parquet(batter_out, args.batter_output)
    batter_out.to_parquet(args.batter_output, index=False)
    print(f"Wrote {len(batter_out):,} rows to {args.batter_output}")


if __name__ == "__main__":
    main()

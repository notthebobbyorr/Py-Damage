from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

WHIFF_NUM_FEATURES = [
    "avg_release_z", "avg_release_x", "avg_ext", "pitch_velo", "rpm", "vbreak",
    "hbreak", "axis", "spin_efficiency", "z_angle_release", "x_angle_release",
    "vaa", "haa", "primary_velo", "primary_loc_adj_vaa", "primary_z_release",
    "primary_x_release", "primary_rpm", "primary_axis", "x", "z",
]
WHIFF_CAT_FEATURES = ["balls", "strikes", "throws", "stands"]
WHIFF_BASE_NUM_FEATURES = [feat for feat in WHIFF_NUM_FEATURES if feat not in {"x", "z"}]
GROUP_PITCHER = ["pitcher_mlbid", "pitcher_name", "season", "level_id", "game_type_group"]
GROUP_PITCH_TYPE = ["pitcher_mlbid", "pitcher_name", "season", "level_id", "game_type_group", "pitch_tag"]

GAME_TYPE_GROUP_MAP = {"S": "Spring Training", "F": "Postseason", "D": "Postseason", "L": "Postseason", "W": "Postseason"}


def _upsert_parquet(new_df: pd.DataFrame, out_path: Path, key_col: str = "season") -> pd.DataFrame:
    """Return new_df merged with any existing rows whose key_col values are NOT in new_df.

    This prevents a daily refresh (run on a single season's accumulator) from
    wiping historical rows that were written by the backfill.
    """
    if out_path.exists():
        existing = pd.read_parquet(out_path)
        new_keys = set(new_df[key_col].dropna().unique())
        existing = existing[~existing[key_col].isin(new_keys)]
        return pd.concat([existing, new_df], ignore_index=True)
    return new_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build level-aware p(swstr) source tables from pitch-level data.")
    parser.add_argument("--parquet-path", type=Path, required=True, help="Path to input pitch-level parquet file.")
    parser.add_argument("--loc-model", type=Path, default=Path(__file__).resolve().parent.parent / "models" / "is_swing_catboost_model_with_locations.cbm")
    parser.add_argument("--base-model", type=Path, default=Path(__file__).resolve().parent.parent / "models" / "is_swing_catboost_model.cbm")
    parser.add_argument("--whiff-loc-model", type=Path, default=Path(__file__).resolve().parent.parent / "models" / "p_whiff_v2_with_locations.cbm")
    parser.add_argument("--whiff-base-model", type=Path, default=Path(__file__).resolve().parent.parent / "models" / "p_whiff_v2.cbm")
    parser.add_argument("--pitcher-output", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "raw" / "pitcher_p_swstr.parquet")
    parser.add_argument("--pitch-type-output", type=Path, default=Path(__file__).resolve().parent.parent / "data" / "raw" / "pitch_types_p_swstr.parquet")
    parser.add_argument("--batch-size", type=int, default=250000)
    return parser.parse_args()


def _add_game_type_group(df: pd.DataFrame) -> pd.DataFrame:
    def _map(gt):
        return GAME_TYPE_GROUP_MAP.get(gt, "Regular Season")
    if "game_type" not in df.columns:
        df["game_type_group"] = "Regular Season"
    else:
        df["game_type_group"] = df["game_type"].map(_map).fillna("Regular Season")
    return df


def prepare_features(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    X = df[feature_columns].copy()
    for col in WHIFF_CAT_FEATURES:
        if col in X.columns:
            X[col] = X[col].astype("string").fillna("UNK")
    return X


def predict_in_batches(model: CatBoostClassifier, df: pd.DataFrame, feature_columns: list[str], batch_size: int) -> np.ndarray:
    preds = np.empty(len(df), dtype=np.float32)
    for start in range(0, len(df), batch_size):
        stop = min(start + batch_size, len(df))
        X = prepare_features(df.iloc[start:stop], feature_columns)
        preds[start:stop] = model.predict_proba(X)[:, 1]
    return preds


def aggregate_outputs(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        df.groupby(group_cols, dropna=False)
        .agg(
            n=("whiff", "size"),
            SwStr_pct=("whiff", lambda s: float(np.mean(s)) * 100.0),
            p_SwStr_pct=("pred_swstr_base", lambda s: float(np.mean(s)) * 100.0),
            p_SwStr_with_loc_pct=("pred_swstr_loc", lambda s: float(np.mean(s)) * 100.0),
            Swing_pct=("swing", lambda s: float(np.mean(s)) * 100.0),
            p_Swing_pct=("pred_swing_base", lambda s: float(np.mean(s)) * 100.0),
            p_Swing_with_loc_pct=("pred_swing_loc", lambda s: float(np.mean(s)) * 100.0),
        )
        .reset_index()
    )


def collapse_duplicates(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if not df.duplicated(group_cols).any():
        return df
    tmp = df.copy()
    for col in ["SwStr_pct", "p_SwStr_pct", "p_SwStr_with_loc_pct", "Swing_pct", "p_Swing_pct", "p_Swing_with_loc_pct"]:
        tmp[col] = tmp[col] * tmp["n"]
    out = tmp.groupby(group_cols, dropna=False, as_index=False).agg(
        n=("n", "sum"),
        SwStr_pct=("SwStr_pct", "sum"),
        p_SwStr_pct=("p_SwStr_pct", "sum"),
        p_SwStr_with_loc_pct=("p_SwStr_with_loc_pct", "sum"),
        Swing_pct=("Swing_pct", "sum"),
        p_Swing_pct=("p_Swing_pct", "sum"),
        p_Swing_with_loc_pct=("p_Swing_with_loc_pct", "sum"),
    )
    for col in ["SwStr_pct", "p_SwStr_pct", "p_SwStr_with_loc_pct", "Swing_pct", "p_Swing_pct", "p_Swing_with_loc_pct"]:
        out[col] = out[col] / out["n"]
    return out


def score_dataframe(
    df: pd.DataFrame,
    swing_loc_model: CatBoostClassifier,
    swing_base_model: CatBoostClassifier,
    whiff_loc_model: CatBoostClassifier,
    whiff_base_model: CatBoostClassifier,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = sorted(set(
        WHIFF_NUM_FEATURES + WHIFF_CAT_FEATURES + [
            "whiff", "swing", "pitcher_role", "pitcher_mlbid", "pitcher_name", "season", "level_id", "pitch_tag",
            "game_type_group",
        ]
    ))
    available = [c for c in required if c in df.columns]
    df = df[available].copy()
    df = df[df["pitcher_role"].isin(["SP", "RP"])].copy()
    df["whiff"] = pd.to_numeric(df["whiff"], errors="coerce").fillna(0).astype("int8")
    df["pitcher_name"] = df["pitcher_name"].astype("string").fillna("UNK")
    df["pitch_tag"] = df["pitch_tag"].astype("string").fillna("UNK")
    for col in WHIFF_CAT_FEATURES:
        df[col] = df[col].astype("string").fillna("UNK")
    df["pred_swing_loc"] = predict_in_batches(swing_loc_model, df, WHIFF_NUM_FEATURES + WHIFF_CAT_FEATURES, batch_size)
    df["pred_swing_base"] = predict_in_batches(swing_base_model, df, WHIFF_BASE_NUM_FEATURES + WHIFF_CAT_FEATURES, batch_size)
    df["pred_whiff_loc_v2"] = predict_in_batches(whiff_loc_model, df, WHIFF_NUM_FEATURES + WHIFF_CAT_FEATURES, batch_size)
    df["pred_whiff_base_v2"] = predict_in_batches(whiff_base_model, df, WHIFF_BASE_NUM_FEATURES + WHIFF_CAT_FEATURES, batch_size)
    df["pred_swstr_loc"] = df["pred_swing_loc"] * df["pred_whiff_loc_v2"]
    df["pred_swstr_base"] = df["pred_swing_base"] * df["pred_whiff_base_v2"]
    return aggregate_outputs(df, GROUP_PITCHER), aggregate_outputs(df, GROUP_PITCH_TYPE)


def main() -> None:
    args = parse_args()
    loc_model = CatBoostClassifier()
    loc_model.load_model(args.loc_model.as_posix())
    base_model = CatBoostClassifier()
    base_model.load_model(args.base_model.as_posix())
    whiff_loc_model = CatBoostClassifier()
    whiff_loc_model.load_model(args.whiff_loc_model.as_posix())
    whiff_base_model = CatBoostClassifier()
    whiff_base_model.load_model(args.whiff_base_model.as_posix())

    if not args.parquet_path.exists():
        raise FileNotFoundError(f"Input parquet not found: {args.parquet_path}")

    print(f"Scoring {args.parquet_path} ...")
    df = pd.read_parquet(args.parquet_path)
    df = _add_game_type_group(df)

    pitcher_out, pitch_type_out = score_dataframe(
        df, loc_model, base_model, whiff_loc_model, whiff_base_model, args.batch_size
    )
    pitcher_out = collapse_duplicates(pitcher_out, GROUP_PITCHER)
    pitch_type_out = collapse_duplicates(pitch_type_out, GROUP_PITCH_TYPE)

    args.pitcher_output.parent.mkdir(parents=True, exist_ok=True)
    args.pitch_type_output.parent.mkdir(parents=True, exist_ok=True)
    pitcher_out = _upsert_parquet(pitcher_out, args.pitcher_output)
    pitch_type_out = _upsert_parquet(pitch_type_out, args.pitch_type_output)
    pitcher_out.to_parquet(args.pitcher_output, index=False)
    pitch_type_out.to_parquet(args.pitch_type_output, index=False)
    pitcher_out.to_csv(args.pitcher_output.with_suffix(".csv"), index=False)
    pitch_type_out.to_csv(args.pitch_type_output.with_suffix(".csv"), index=False)
    print(f"Wrote {len(pitcher_out):,} rows to {args.pitcher_output}")
    print(f"Wrote {len(pitch_type_out):,} rows to {args.pitch_type_output}")


if __name__ == "__main__":
    main()

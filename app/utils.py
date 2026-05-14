from __future__ import annotations

import numpy as np
import pandas as pd

from app.config import (
    HITTER_HIGHER_IS_WORSE_METRICS,
    HITTER_MLB_DIRECTION_MAP,
    HITTER_MLB_MIN_SHIFT_FLOOR,
    HITTER_MLB_MIN_SHIFT_SCALE,
    HITTER_MLB_MIN_SHIFT_SCALE_OVERRIDES,
    LEVEL_LABELS,
    PITCHER_MLB_DIRECTION_MAP,
    PITCHER_MLB_MIN_SHIFT_SCALE,
    PITCHER_MLB_PASS_THROUGH_COLS,
)


# ---------------------------------------------------------------------------
# Column normalization helpers
# ---------------------------------------------------------------------------

def _normalize_team_col(df: pd.DataFrame, old_col: str) -> pd.DataFrame:
    """If 'team' column exists, rename it to old_col for backward compatibility."""
    if df.empty:
        return df
    if "team" in df.columns and old_col not in df.columns:
        return df.rename(columns={"team": old_col})
    return df


def _normalize_la_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Rename old FB_pct/GB_pct columns to new LA_gte_20/LA_lte_0 names."""
    if df.empty:
        return df
    rename_map = {}
    if "FB_pct" in df.columns and "LA_gte_20" not in df.columns:
        rename_map["FB_pct"] = "LA_gte_20"
    if "GB_pct" in df.columns and "LA_lte_0" not in df.columns:
        rename_map["GB_pct"] = "LA_lte_0"
    if rename_map:
        return df.rename(columns=rename_map)
    return df


def _normalize_split_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for col in ["split_type", "split"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def _similarity_choice_labels(
    df: pd.DataFrame,
    display_map: dict[str, str],
    exclude_cols: set[str],
) -> tuple[list[str], dict[str, str]]:
    if df.empty:
        return [], {}
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    filtered = []
    for col in numeric_cols:
        if col in exclude_cols:
            continue
        if col == "reg_prop":
            continue
        if col.endswith("_raw") or col.endswith("_raw_reg"):
            continue
        if "_num" in col or "_den" in col or "_n" in col:
            continue
        if col.endswith("_id") or col.endswith("_mlbid"):
            continue
        filtered.append(col)
    labels = {}
    for col in filtered:
        label = display_map.get(col, col.replace("_", " ").title())
        if label.endswith(" Reg"):
            label = label[: -len(" Reg")]
        labels[col] = label
    return filtered, labels


# ---------------------------------------------------------------------------
# Regression / display map helpers
# ---------------------------------------------------------------------------

def _merge_regressed(
    base_df: pd.DataFrame, reg_df: pd.DataFrame, keys: list[str]
) -> pd.DataFrame:
    if base_df.empty or reg_df.empty:
        return pd.DataFrame()
    reg_cols = [
        c
        for c in reg_df.columns
        if c.endswith("_reg") or c.endswith("_raw") or c.endswith("_n")
    ]
    keep_cols = list(dict.fromkeys(keys + reg_cols))
    reg_small = reg_df[keep_cols].drop_duplicates(subset=keys)
    # Normalize key dtypes before merge to avoid pandas factorizer crashes
    # with mixed/downcast integer dtypes (e.g. int8/int16/int32).
    left = base_df.copy()
    right = reg_small.copy()
    for key in keys:
        if key not in left.columns or key not in right.columns:
            continue
        left_key = left[key]
        right_key = right[key]
        if pd.api.types.is_numeric_dtype(left_key) or pd.api.types.is_numeric_dtype(
            right_key
        ):
            left[key] = pd.to_numeric(left_key, errors="coerce").astype("Int64")
            right[key] = pd.to_numeric(right_key, errors="coerce").astype("Int64")
        else:
            left[key] = left_key.astype("string")
            right[key] = right_key.astype("string")
    return left.merge(right, on=keys, how="left")


def _hitter_display_map(include_mlb_eq: bool = False) -> dict[str, str]:
    display_map = {
        "hitter_name": "Name",
        "hitting_code": "Team",
        "season": "Season",
        "bbe": "BBE",
        "damage_rate_reg": "Damage/BBE (%)",
        "EV90th_reg": "90th Pctile EV",
        "pull_FB_pct_reg": "Pulled FB (%)",
        "selection_skill_reg": "Selectivity (%)",
        "hittable_pitches_taken_reg": "Hittable Pitch Take (%)",
        "chase_reg": "Chase (%)",
        "z_con_reg": "Z-Contact (%)",
        "secondary_whiff_pct_reg": "Whiff vs. Secondaries (%)",
        "similarity_score": "Similarity (0-100)",
        "LA_gte_20_reg": "LA>=20 (%)",
        "LA_lte_0_reg": "LA<=0%",
        "SEAGER_reg": "SEAGER",
        "whiffs_vs_95_reg": "Whiff vs. 95+ (%)",
        "contact_vs_avg_reg": "Contact Over Expected (%)",
        "LD_pct_reg": "0<LA<20 (%)",
        "bat_speed_reg": "Avg Swing Speed",
        "fast_swing_pct": "Fast Swing (%)",
        "swing_length_reg": "Swing Length",
        "attack_angle_reg": "Attack Angle",
        "attack_direction": "Attack Direction",
        "intercept_x_inches": "Intercept X (in.)",
        "intercept_y_inches": "Intercept Y (in.)",
        "swing_path_tilt_reg": "VBA",
        "max_EV_reg": "Max EV",
        "takeoff_rate_reg": "Takeoff%",
    }
    if not include_mlb_eq:
        return display_map
    eq_map = {}
    for col, label in display_map.items():
        if col.endswith("_reg"):
            eq_map[f"{col}_mlb_eq"] = f"{label} MLB Eq"
    return {**display_map, **eq_map}


def _pitcher_display_map(include_mlb_eq: bool = False) -> dict[str, str]:
    display_map = {
        "name": "Name",
        "pitching_code": "Team",
        "season": "Season",
        "stuff": "Pitch Grade",
        "grade_v13": "Execution Grade",
        "stuff_raw_reg": "Pitch Grade (Raw Model)",
        "fastball_velo_reg": "FA mph",
        "max_velo_reg": "Max FA mph",
        "fastball_vaa_reg": "FA VAA",
        "loc_adj_vaa_reg": "Loc-Adj VAA",
        "SwStr_reg": "SwStr (%)",
        "Ball_pct_reg": "Ball (%)",
        "Chase_reg": "Chase (%)",
        "Z_Contact_reg": "Z-Contact (%)",
        "Zone_reg": "Zone (%)",
        "CSW_reg": "CSW (%)",
        "pWhiff_reg": "pWhiff (%)",
        "FA_pct_reg": "FA (%)",
        "BB_rpm_reg": "BB RPM",
        "FA_spin_eff_reg": "FA Spin Efficiency (%)",
        "LA_lte_0_reg": "LA<=0%",
        "LD_pct_reg": "0<LA<20 (%)",
        "LA_gte_20_reg": "LA>=20 (%)",
        "rel_z_reg": "Vertical Release (ft.)",
        "rel_x_reg": "Horizontal Release (ft.)",
        "ext_reg": "Extension (ft.)",
        "inf_arm_angle": "Inferred Arm Angle",
        "takeoff_rate_reg": "Takeoff% Against",
        "similarity_score": "Similarity (0-100)",
    }
    if not include_mlb_eq:
        return display_map
    eq_map: dict[str, str] = {}
    for col, label in display_map.items():
        if col.endswith("_reg") or col == "stuff":
            eq_map[f"{col}_mlb_eq"] = f"{label} MLB Eq"
    return {**display_map, **eq_map}


def _pitch_display_map() -> dict[str, str]:
    return {
        "name": "Name",
        "pitching_code": "Team",
        "season": "Season",
        "pitch_tag": "Pitch Type",
        "pitches": "#",
        "pct": "Usage (%)",
        "stuff": "Pitch Grade",
        "grade_v13": "Execution Grade",
        "velo": "Velo",
        "max_velo": "Max Velo",
        "vaa": "VAA",
        "haa": "HAA",
        "vbreak": "IVB (in.)",
        "hbreak": "HB (in.)",
        "rel_z": "Vertical Release (ft.)",
        "rel_x": "Horizontal Release (ft.)",
        "ext": "Extension (ft.)",
        "z_angle_release": "VRA",
        "x_angle_release": "HRA",
        "inf_arm_angle": "Inferred Arm Angle",
        "SwStr": "SwStr (%)",
        "Zone": "Zone (%)",
        "Chase": "Chase (%)",
        "Ball_pct": "Ball (%)",
        "Z_Contact": "Z-Contact (%)",
        "CSW": "CSW (%)",
        "similarity_score": "Similarity (0-100)",
    }


# ---------------------------------------------------------------------------
# MLB equivalency computation
# ---------------------------------------------------------------------------

def _compose_linear(a1: float, b1: float, a2: float, b2: float) -> tuple[float, float]:
    return a2 + (b2 * a1), b2 * b1


def _weighted_linear_fit(
    x: np.ndarray, y: np.ndarray, w: np.ndarray
) -> tuple[float, float, int]:
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    n = int(valid.sum())
    if n < 2:
        return np.nan, np.nan, n
    xv = x[valid]
    yv = y[valid]
    wv = w[valid]
    design = np.column_stack([np.ones(len(xv)), xv])
    sqrt_w = np.sqrt(wv)
    try:
        coef, *_ = np.linalg.lstsq(design * sqrt_w[:, None], yv * sqrt_w, rcond=None)
    except Exception:
        return np.nan, np.nan, n
    return float(coef[0]), float(coef[1]), n


def _build_hitter_mlb_equivalencies(
    base_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if base_df.empty:
        return pd.DataFrame(), pd.DataFrame(), []
    if "PA" not in base_df.columns:
        return base_df.copy(), pd.DataFrame(), []

    metric_cols = [
        col
        for col in base_df.select_dtypes(include="number").columns
        if col.endswith("_reg") and col != "reg_prop"
    ]
    if not metric_cols:
        return base_df.copy(), pd.DataFrame(), []

    keys = ["batter_mlbid", "season", "level_id"]
    base_cols = [col for col in keys + ["PA"] + metric_cols if col in base_df.columns]
    fit_df = base_df[base_cols].copy()
    grouped = fit_df.groupby(keys, as_index=False).agg(
        {"PA": "sum", **{col: "mean" for col in metric_cols}}
    )

    means = grouped.groupby(["season", "level_id"])[metric_cols].mean()
    stds = grouped.groupby(["season", "level_id"])[metric_cols].std(ddof=0)
    means = means.add_suffix("__mean")
    stds = stds.add_suffix("__std")
    moments = means.join(stds).reset_index()

    z_df = grouped.merge(moments, on=["season", "level_id"], how="left")
    for col in metric_cols:
        std_col = f"{col}__std"
        z_df[f"{col}__z"] = (z_df[col] - z_df[f"{col}__mean"]) / z_df[std_col].replace(
            0, np.nan
        )

    src = z_df.rename(
        columns={
            "season": "src_season",
            "level_id": "src_level",
            "PA": "src_PA",
            **{f"{col}__z": f"src_{col}__z" for col in metric_cols},
        }
    )
    dst = z_df.rename(
        columns={
            "season": "dst_season",
            "level_id": "dst_level",
            "PA": "dst_PA",
            **{f"{col}__z": f"dst_{col}__z" for col in metric_cols},
        }
    )
    same_pairs = src.merge(
        dst,
        left_on=["batter_mlbid", "src_season"],
        right_on=["batter_mlbid", "dst_season"],
        how="inner",
    ).assign(pair_group="same_season")
    dst_next = dst.copy()
    dst_next["src_season"] = dst_next["dst_season"] - 1
    next_pairs = src.merge(
        dst_next,
        on=["batter_mlbid", "src_season"],
        how="inner",
    ).assign(pair_group="next_season")
    pairs = pd.concat([same_pairs, next_pairs], ignore_index=True, sort=False)
    pairs = pairs[pairs["src_level"] > pairs["dst_level"]]

    edge_thresholds: dict[tuple[int, int], tuple[int, int]] = {
        (11, 1): (50, 50),
        (14, 11): (50, 50),
        (16, 14): (10, 10),
    }
    prior_a = 0.0
    prior_b = 0.5
    shrink_k = 50.0
    edge_coeff: dict[tuple[int, int, str], tuple[float, float, int]] = {}
    coeff_rows: list[dict[str, object]] = []

    for (src_level, dst_level), (min_src_pa, min_dst_pa) in edge_thresholds.items():
        edge_df = pairs[
            (pairs["src_level"] == src_level)
            & (pairs["dst_level"] == dst_level)
            & (pairs["src_PA"] >= min_src_pa)
            & (pairs["dst_PA"] >= min_dst_pa)
        ].copy()
        if edge_df.empty:
            for col in metric_cols:
                edge_coeff[(src_level, dst_level, col)] = (prior_a, prior_b, 0)
                coeff_rows.append(
                    {
                        "src_level": src_level,
                        "dst_level": dst_level,
                        "metric": col,
                        "a": prior_a,
                        "b": prior_b,
                        "n": 0,
                        "min_src_pa": min_src_pa,
                        "min_dst_pa": min_dst_pa,
                        "fit_type": "intra+inter-season",
                    }
                )
            continue

        weights = np.sqrt(
            np.clip(edge_df["src_PA"].to_numpy(dtype=float), 0, None)
            * np.clip(edge_df["dst_PA"].to_numpy(dtype=float), 0, None)
        )
        for col in metric_cols:
            x = edge_df[f"src_{col}__z"].to_numpy(dtype=float)
            y = edge_df[f"dst_{col}__z"].to_numpy(dtype=float)
            raw_a, raw_b, n = _weighted_linear_fit(x, y, weights)
            if not np.isfinite(raw_a):
                raw_a = prior_a
            if not np.isfinite(raw_b):
                raw_b = prior_b
            reliability = n / (n + shrink_k) if n > 0 else 0.0
            fit_a = reliability * raw_a + (1.0 - reliability) * prior_a
            fit_b = reliability * raw_b + (1.0 - reliability) * prior_b
            fit_a = float(np.clip(fit_a, -1.5, 1.5))
            fit_b = float(np.clip(fit_b, -0.25, 1.25))
            edge_coeff[(src_level, dst_level, col)] = (fit_a, fit_b, n)
            coeff_rows.append(
                {
                    "src_level": src_level,
                    "dst_level": dst_level,
                    "metric": col,
                    "a": fit_a,
                    "b": fit_b,
                    "n": n,
                    "min_src_pa": min_src_pa,
                    "min_dst_pa": min_dst_pa,
                    "fit_type": "intra+inter-season",
                }
            )

    level_mlb_coeff: dict[tuple[int, str], tuple[float, float, int]] = {}
    for col in metric_cols:
        a11, b11, n11 = edge_coeff[(11, 1, col)]
        a14, b14, n14 = edge_coeff[(14, 11, col)]
        a16, b16, n16 = edge_coeff[(16, 14, col)]
        a14_to_1, b14_to_1 = _compose_linear(a14, b14, a11, b11)
        a16_to_11, b16_to_11 = _compose_linear(a16, b16, a14, b14)
        a16_to_1, b16_to_1 = _compose_linear(a16_to_11, b16_to_11, a11, b11)
        level_mlb_coeff[(1, col)] = (0.0, 1.0, n11)
        level_mlb_coeff[(11, col)] = (a11, b11, n11)
        level_mlb_coeff[(14, col)] = (a14_to_1, b14_to_1, min(n14, n11))
        level_mlb_coeff[(16, col)] = (a16_to_1, b16_to_1, min(n16, n14, n11))
        coeff_rows.extend(
            [
                {
                    "src_level": 14,
                    "dst_level": 1,
                    "metric": col,
                    "a": a14_to_1,
                    "b": b14_to_1,
                    "n": min(n14, n11),
                    "min_src_pa": 50,
                    "min_dst_pa": 50,
                    "fit_type": "chained",
                },
                {
                    "src_level": 16,
                    "dst_level": 1,
                    "metric": col,
                    "a": a16_to_1,
                    "b": b16_to_1,
                    "n": min(n16, n14, n11),
                    "min_src_pa": 10,
                    "min_dst_pa": 50,
                    "fit_type": "chained",
                },
            ]
        )

    mlb_moments = moments[moments["level_id"] == 1].drop(columns=["level_id"]).copy()
    mlb_moments = mlb_moments.rename(
        columns={
            f"{col}__mean": f"{col}__mlb_mean"
            for col in metric_cols
            if f"{col}__mean" in mlb_moments.columns
        }
        | {
            f"{col}__std": f"{col}__mlb_std"
            for col in metric_cols
            if f"{col}__std" in mlb_moments.columns
        }
    )

    out = base_df.copy()
    out = out.merge(moments, on=["season", "level_id"], how="left")
    out = out.merge(mlb_moments, on=["season"], how="left")

    for col in metric_cols:
        src_mean_col = f"{col}__mean"
        src_std_col = f"{col}__std"
        mlb_mean_col = f"{col}__mlb_mean"
        mlb_std_col = f"{col}__mlb_std"
        if (
            col not in out.columns
            or src_mean_col not in out.columns
            or src_std_col not in out.columns
            or mlb_mean_col not in out.columns
            or mlb_std_col not in out.columns
        ):
            continue
        src_z = (out[col] - out[src_mean_col]) / out[src_std_col].replace(0, np.nan)
        a_map = {
            level_id: level_mlb_coeff.get((level_id, col), (prior_a, prior_b, 0))[0]
            for level_id in LEVEL_LABELS
        }
        b_map = {
            level_id: level_mlb_coeff.get((level_id, col), (prior_a, prior_b, 0))[1]
            for level_id in LEVEL_LABELS
        }
        pred_z = out["level_id"].map(a_map) + (out["level_id"].map(b_map) * src_z)
        pred = out[mlb_mean_col] + (pred_z * out[mlb_std_col])
        mlb_mask = out["level_id"] == 1
        non_mlb_mask = ~mlb_mask
        if col in HITTER_HIGHER_IS_WORSE_METRICS:
            pred.loc[non_mlb_mask] = np.maximum(
                pred.loc[non_mlb_mask],
                out.loc[non_mlb_mask, col],
            )
        else:
            pred.loc[non_mlb_mask] = np.minimum(
                pred.loc[non_mlb_mask],
                out.loc[non_mlb_mask, col],
            )
        direction = HITTER_MLB_DIRECTION_MAP.get(col)
        if direction in {"up", "down"}:
            shift_scale = HITTER_MLB_MIN_SHIFT_SCALE_OVERRIDES.get(
                col, HITTER_MLB_MIN_SHIFT_SCALE
            )
            shift = (
                (out[src_mean_col] - out[mlb_mean_col]).abs() * shift_scale
            ).fillna(0.0)
            shift_floor = HITTER_MLB_MIN_SHIFT_FLOOR.get(col, 0.0)
            if shift_floor > 0:
                shift = shift.clip(lower=shift_floor)
            if direction == "up":
                pred.loc[non_mlb_mask] = np.maximum(
                    pred.loc[non_mlb_mask],
                    out.loc[non_mlb_mask, col] + shift.loc[non_mlb_mask],
                )
            else:
                pred.loc[non_mlb_mask] = np.minimum(
                    pred.loc[non_mlb_mask],
                    out.loc[non_mlb_mask, col] - shift.loc[non_mlb_mask],
                )
        pred.loc[mlb_mask] = out.loc[mlb_mask, col]
        out[f"{col}_mlb_eq"] = pred

    helper_cols: list[str] = []
    for col in metric_cols:
        helper_cols.extend(
            [f"{col}__mean", f"{col}__std", f"{col}__mlb_mean", f"{col}__mlb_std"]
        )
    out = out.drop(columns=[col for col in helper_cols if col in out.columns])

    coeff_df = pd.DataFrame(coeff_rows)
    return out, coeff_df, metric_cols


def _build_pitcher_mlb_equivalencies(
    base_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if base_df.empty:
        return pd.DataFrame(), pd.DataFrame(), []
    if "TBF" not in base_df.columns:
        return base_df.copy(), pd.DataFrame(), []

    metric_cols = [
        col
        for col in base_df.select_dtypes(include="number").columns
        if col.endswith("_reg")
        and col != "reg_prop"
        and col not in PITCHER_MLB_PASS_THROUGH_COLS
    ]
    if not metric_cols:
        out = base_df.copy()
        for col in PITCHER_MLB_PASS_THROUGH_COLS:
            if col in out.columns:
                out[f"{col}_mlb_eq"] = out[col]
        pass_through_metrics = [
            col
            for col in PITCHER_MLB_PASS_THROUGH_COLS
            if f"{col}_mlb_eq" in out.columns
        ]
        return out, pd.DataFrame(), pass_through_metrics

    keys = ["pitcher_mlbid", "season", "level_id"]
    base_cols = [col for col in keys + ["TBF"] + metric_cols if col in base_df.columns]
    fit_df = base_df[base_cols].copy()
    grouped = fit_df.groupby(keys, as_index=False).agg(
        {"TBF": "sum", **{col: "mean" for col in metric_cols}}
    )

    means = grouped.groupby(["season", "level_id"])[metric_cols].mean()
    stds = grouped.groupby(["season", "level_id"])[metric_cols].std(ddof=0)
    means = means.add_suffix("__mean")
    stds = stds.add_suffix("__std")
    moments = means.join(stds).reset_index()

    z_df = grouped.merge(moments, on=["season", "level_id"], how="left")
    for col in metric_cols:
        std_col = f"{col}__std"
        z_df[f"{col}__z"] = (z_df[col] - z_df[f"{col}__mean"]) / z_df[std_col].replace(
            0, np.nan
        )

    src = z_df.rename(
        columns={
            "season": "src_season",
            "level_id": "src_level",
            "TBF": "src_TBF",
            **{f"{col}__z": f"src_{col}__z" for col in metric_cols},
        }
    )
    dst = z_df.rename(
        columns={
            "season": "dst_season",
            "level_id": "dst_level",
            "TBF": "dst_TBF",
            **{f"{col}__z": f"dst_{col}__z" for col in metric_cols},
        }
    )
    same_pairs = src.merge(
        dst,
        left_on=["pitcher_mlbid", "src_season"],
        right_on=["pitcher_mlbid", "dst_season"],
        how="inner",
    ).assign(pair_group="same_season")
    dst_next = dst.copy()
    dst_next["src_season"] = dst_next["dst_season"] - 1
    next_pairs = src.merge(
        dst_next,
        on=["pitcher_mlbid", "src_season"],
        how="inner",
    ).assign(pair_group="next_season")
    pairs = pd.concat([same_pairs, next_pairs], ignore_index=True, sort=False)
    pairs = pairs[pairs["src_level"] > pairs["dst_level"]]

    edge_thresholds: dict[tuple[int, int], tuple[int, int]] = {
        (11, 1): (60, 60),
        (14, 11): (60, 60),
        (16, 14): (60, 60),
    }
    prior_a = 0.0
    prior_b = 0.5
    shrink_k = 50.0
    edge_coeff: dict[tuple[int, int, str], tuple[float, float, int]] = {}
    coeff_rows: list[dict[str, object]] = []

    for (src_level, dst_level), (min_src_tbf, min_dst_tbf) in edge_thresholds.items():
        edge_df = pairs[
            (pairs["src_level"] == src_level)
            & (pairs["dst_level"] == dst_level)
            & (pairs["src_TBF"] >= min_src_tbf)
            & (pairs["dst_TBF"] >= min_dst_tbf)
        ].copy()
        if edge_df.empty:
            for col in metric_cols:
                edge_coeff[(src_level, dst_level, col)] = (prior_a, prior_b, 0)
                coeff_rows.append(
                    {
                        "src_level": src_level,
                        "dst_level": dst_level,
                        "metric": col,
                        "a": prior_a,
                        "b": prior_b,
                        "n": 0,
                        "min_src_tbf": min_src_tbf,
                        "min_dst_tbf": min_dst_tbf,
                        "fit_type": "intra+inter-season",
                    }
                )
            continue

        weights = np.sqrt(
            np.clip(edge_df["src_TBF"].to_numpy(dtype=float), 0, None)
            * np.clip(edge_df["dst_TBF"].to_numpy(dtype=float), 0, None)
        )
        for col in metric_cols:
            x = edge_df[f"src_{col}__z"].to_numpy(dtype=float)
            y = edge_df[f"dst_{col}__z"].to_numpy(dtype=float)
            raw_a, raw_b, n = _weighted_linear_fit(x, y, weights)
            if not np.isfinite(raw_a):
                raw_a = prior_a
            if not np.isfinite(raw_b):
                raw_b = prior_b
            reliability = n / (n + shrink_k) if n > 0 else 0.0
            fit_a = reliability * raw_a + (1.0 - reliability) * prior_a
            fit_b = reliability * raw_b + (1.0 - reliability) * prior_b
            fit_a = float(np.clip(fit_a, -1.5, 1.5))
            fit_b = float(np.clip(fit_b, -0.25, 1.25))
            edge_coeff[(src_level, dst_level, col)] = (fit_a, fit_b, n)
            coeff_rows.append(
                {
                    "src_level": src_level,
                    "dst_level": dst_level,
                    "metric": col,
                    "a": fit_a,
                    "b": fit_b,
                    "n": n,
                    "min_src_tbf": min_src_tbf,
                    "min_dst_tbf": min_dst_tbf,
                    "fit_type": "intra+inter-season",
                }
            )

    level_mlb_coeff: dict[tuple[int, str], tuple[float, float, int]] = {}
    for col in metric_cols:
        a11, b11, n11 = edge_coeff[(11, 1, col)]
        a14, b14, n14 = edge_coeff[(14, 11, col)]
        a16, b16, n16 = edge_coeff[(16, 14, col)]
        a14_to_1, b14_to_1 = _compose_linear(a14, b14, a11, b11)
        a16_to_11, b16_to_11 = _compose_linear(a16, b16, a14, b14)
        a16_to_1, b16_to_1 = _compose_linear(a16_to_11, b16_to_11, a11, b11)
        level_mlb_coeff[(1, col)] = (0.0, 1.0, n11)
        level_mlb_coeff[(11, col)] = (a11, b11, n11)
        level_mlb_coeff[(14, col)] = (a14_to_1, b14_to_1, min(n14, n11))
        level_mlb_coeff[(16, col)] = (a16_to_1, b16_to_1, min(n16, n14, n11))
        coeff_rows.extend(
            [
                {
                    "src_level": 14,
                    "dst_level": 1,
                    "metric": col,
                    "a": a14_to_1,
                    "b": b14_to_1,
                    "n": min(n14, n11),
                    "min_src_tbf": 60,
                    "min_dst_tbf": 60,
                    "fit_type": "chained",
                },
                {
                    "src_level": 16,
                    "dst_level": 1,
                    "metric": col,
                    "a": a16_to_1,
                    "b": b16_to_1,
                    "n": min(n16, n14, n11),
                    "min_src_tbf": 60,
                    "min_dst_tbf": 60,
                    "fit_type": "chained",
                },
            ]
        )

    mlb_moments = moments[moments["level_id"] == 1].drop(columns=["level_id"]).copy()
    mlb_moments = mlb_moments.rename(
        columns={
            f"{col}__mean": f"{col}__mlb_mean"
            for col in metric_cols
            if f"{col}__mean" in mlb_moments.columns
        }
        | {
            f"{col}__std": f"{col}__mlb_std"
            for col in metric_cols
            if f"{col}__std" in mlb_moments.columns
        }
    )

    out = base_df.copy()
    out = out.merge(moments, on=["season", "level_id"], how="left")
    out = out.merge(mlb_moments, on=["season"], how="left")

    for col in metric_cols:
        src_mean_col = f"{col}__mean"
        src_std_col = f"{col}__std"
        mlb_mean_col = f"{col}__mlb_mean"
        mlb_std_col = f"{col}__mlb_std"
        if (
            col not in out.columns
            or src_mean_col not in out.columns
            or src_std_col not in out.columns
            or mlb_mean_col not in out.columns
            or mlb_std_col not in out.columns
        ):
            continue
        src_z = (out[col] - out[src_mean_col]) / out[src_std_col].replace(0, np.nan)
        a_map = {
            level_id: level_mlb_coeff.get((level_id, col), (prior_a, prior_b, 0))[0]
            for level_id in LEVEL_LABELS
        }
        b_map = {
            level_id: level_mlb_coeff.get((level_id, col), (prior_a, prior_b, 0))[1]
            for level_id in LEVEL_LABELS
        }
        pred_z = out["level_id"].map(a_map) + (out["level_id"].map(b_map) * src_z)
        pred = out[mlb_mean_col] + (pred_z * out[mlb_std_col])
        mlb_mask = out["level_id"] == 1
        non_mlb_mask = ~mlb_mask
        direction = PITCHER_MLB_DIRECTION_MAP.get(col)
        if direction == "up":
            pred.loc[non_mlb_mask] = np.maximum(
                pred.loc[non_mlb_mask],
                out.loc[non_mlb_mask, col],
            )
            shift = (
                (out[src_mean_col] - out[mlb_mean_col]).abs()
                * PITCHER_MLB_MIN_SHIFT_SCALE
            ).fillna(0.0)
            pred.loc[non_mlb_mask] = np.maximum(
                pred.loc[non_mlb_mask],
                out.loc[non_mlb_mask, col] + shift.loc[non_mlb_mask],
            )
        elif direction == "down":
            pred.loc[non_mlb_mask] = np.minimum(
                pred.loc[non_mlb_mask],
                out.loc[non_mlb_mask, col],
            )
            shift = (
                (out[src_mean_col] - out[mlb_mean_col]).abs()
                * PITCHER_MLB_MIN_SHIFT_SCALE
            ).fillna(0.0)
            pred.loc[non_mlb_mask] = np.minimum(
                pred.loc[non_mlb_mask],
                out.loc[non_mlb_mask, col] - shift.loc[non_mlb_mask],
            )
        pred.loc[mlb_mask] = out.loc[mlb_mask, col]
        out[f"{col}_mlb_eq"] = pred

    for col in PITCHER_MLB_PASS_THROUGH_COLS:
        if col in out.columns:
            out[f"{col}_mlb_eq"] = out[col]

    helper_cols: list[str] = []
    for col in metric_cols:
        helper_cols.extend(
            [f"{col}__mean", f"{col}__std", f"{col}__mlb_mean", f"{col}__mlb_std"]
        )
    out = out.drop(columns=[col for col in helper_cols if col in out.columns])

    coeff_df = pd.DataFrame(coeff_rows)
    all_mlb_eq_metrics = metric_cols + [
        col for col in PITCHER_MLB_PASS_THROUGH_COLS if f"{col}_mlb_eq" in out.columns
    ]
    all_mlb_eq_metrics = list(dict.fromkeys(all_mlb_eq_metrics))
    return out, coeff_df, all_mlb_eq_metrics


def maybe_add_level_col(df: pd.DataFrame, level: str) -> pd.DataFrame:
    """Insert a readable 'Level' column after 'Team' when all levels are shown.

    Only inserts when level == 'All' and __level is present. Positions the
    column immediately after 'Team' when Team exists, otherwise after 'Name'.
    """
    if level != "All" or "__level" not in df.columns:
        return df
    level_series = df["__level"].map(
        lambda v: LEVEL_LABELS.get(int(v), str(int(v)))
    )
    df = df.copy()
    if "Team" in df.columns:
        idx = df.columns.get_loc("Team") + 1
    elif "Name" in df.columns:
        idx = df.columns.get_loc("Name") + 1
    else:
        idx = 0
    df.insert(idx, "Level", level_series)
    return df


def rank_for_display(
    df: pd.DataFrame,
    cols: list[str],
    group_cols: list[str],
    reverse_cols: set[str] | None = None,
) -> pd.DataFrame:
    """Compute 1-100 percentile ranks for raw columns within each group.

    Adds a ``<col>_pctile`` column for every ``col`` that exists in ``df``.
    Columns in ``reverse_cols`` are ranked ascending=False so lower raw value
    → higher percentile (rank 95 = best). Safe when a column is all-NaN.
    """
    if reverse_cols is None:
        reverse_cols = set()
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        pctile_col = f"{col}_pctile"
        ascending = col not in reverse_cols
        try:
            ranked = (
                df.groupby(group_cols, observed=True)[col]
                .rank(pct=True, na_option="keep", ascending=ascending)
                .mul(100)
                .round()
                .clip(1, 100)
            )
            df[pctile_col] = pd.to_numeric(ranked, errors="coerce").astype("Int64")
        except Exception:
            df[pctile_col] = pd.array([pd.NA] * len(df), dtype="Int64")
    return df

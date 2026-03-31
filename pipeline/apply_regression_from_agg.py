from __future__ import annotations

import argparse
import json
from pathlib import Path

_REPO_DIR = Path(__file__).resolve().parent.parent
from typing import Any, Dict, List, Optional

import polars as pl

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None


PERCENT_MEAN_STATS = {
    "pWhiff",
    "pred_whiff_pct",
    "pred_whiff_loc_mean",
    "p_SwStr_pct",
    "p_SwStr_with_loc_pct",
    "Swing_pct",
    "p_Swing_pct",
    "p_Swing_with_loc_pct",
    "Damage_pct",
    "p_Damage_pct",
    "p_Damage_with_loc_pct",
}


def load_config(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if yaml is None:  # pragma: no cover
        raise RuntimeError("PyYAML is required for .yml config files.")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def round_floats(df: pl.DataFrame, places: int = 1) -> pl.DataFrame:
    float_cols = [
        c
        for c, d in zip(df.columns, df.dtypes)
        if d in (pl.Float32, pl.Float64)
    ]
    if not float_cols:
        return df
    return df.with_columns([pl.col(c).round(places) for c in float_cols])


def _join_constants(
    df: pl.DataFrame, constants: pl.DataFrame, dataset: str, stat: str
) -> pl.DataFrame:
    const = constants.filter(
        (pl.col("dataset") == dataset) & (pl.col("stat") == stat)
    )
    if const.is_empty():
        df = df.with_columns(
            [pl.lit(None).alias("mu"), pl.lit(None).alias("K")]
        )
        return df
    join_keys = [
        c
        for c in ["level_id", "pitch_tag", "season"]
        if c in const.columns and c in df.columns
    ]
    if join_keys:
        return df.join(const.select(join_keys + ["mu", "K"]), on=join_keys, how="left")
    return df.join(const.select(["mu", "K"]), how="cross")


def add_league_contact_baseline(df: pl.DataFrame, src: pl.DataFrame) -> pl.DataFrame:
    required = {
        "pred_whiff_loc_mean",
        "pred_whiff_loc_mean_n",
        "whiff_rate_num",
        "whiff_rate_den",
        "season",
        "level_id",
    }
    if not required.issubset(set(src.columns)):
        return df
    group_keys = ["season", "level_id"]
    if "game_type_group" in src.columns:
        group_keys.append("game_type_group")
    league = (
        src.group_by(group_keys)
        .agg(
            [
                (
                    (pl.col("pred_whiff_loc_mean") * pl.col("pred_whiff_loc_mean_n")).sum()
                    / pl.col("pred_whiff_loc_mean_n").sum()
                ).alias("lg_exp_whiff"),
                (
                    pl.col("whiff_rate_num").sum()
                    / pl.col("whiff_rate_den").sum()
                    * 100.0
                ).alias("lg_act_whiff"),
            ]
        )
        .with_columns(
            (pl.col("lg_exp_whiff") - pl.col("lg_act_whiff")).alias(
                "lg_contact_baseline"
            )
        )
        .select(group_keys + ["lg_contact_baseline"])
    )
    join_keys = [c for c in group_keys if c in df.columns]
    if "season" in df.columns and "level_id" in df.columns:
        return df.join(league, on=join_keys, how="left")
    return df


def apply_rate_from_agg(
    df: pl.DataFrame,
    constants: pl.DataFrame,
    dataset: str,
    stat: str,
    key_cols: List[str],
) -> Optional[pl.DataFrame]:
    num_col = f"{stat}_num"
    den_col = f"{stat}_den"
    if num_col not in df.columns or den_col not in df.columns:
        return None
    keep_keys = [c for c in key_cols if c in df.columns]
    out = df.select(keep_keys + [pl.col(num_col).alias("num"), pl.col(den_col).alias("den")])
    out = out.filter(pl.col("den") > 0)
    out = _join_constants(out, constants, dataset, stat)
    raw_prop = pl.col("num") / pl.col("den")
    reg_prop = (pl.col("num") + pl.col("K") * pl.col("mu")) / (
        pl.col("den") + pl.col("K")
    )
    out = out.with_columns(
        pl.when(pl.col("K").is_null() | pl.col("mu").is_null() | (pl.col("K") == 0))
        .then(raw_prop)
        .otherwise(reg_prop)
        .alias("reg_prop")
    ).with_columns(
        [
            (raw_prop * 100.0).alias(f"{stat}_raw"),
            (pl.col("reg_prop") * 100.0).alias(f"{stat}_reg"),
            pl.col("den").alias(f"{stat}_n"),
        ]
    )
    return out.drop(["num", "den", "mu", "K", "reg_prop"])


def _resolve_mean_n_col(df: pl.DataFrame, stat: str) -> Optional[str]:
    candidates = [f"{stat}_n"]
    if stat.endswith("_pct"):
        candidates.append(f"{stat[:-4]}_n")
    if stat.endswith("_with_loc_pct"):
        candidates.append(stat.replace("_with_loc_pct", "_n"))
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def apply_mean_from_agg(
    df: pl.DataFrame,
    constants: pl.DataFrame,
    dataset: str,
    stat: str,
    key_cols: List[str],
) -> Optional[pl.DataFrame]:
    n_col = _resolve_mean_n_col(df, stat)
    if stat not in df.columns or n_col is None:
        return None
    keep_keys = [c for c in key_cols if c in df.columns]
    out = df.select(keep_keys + [pl.col(stat).alias("raw"), pl.col(n_col).alias("n")])
    out = out.with_columns([
        pl.col("raw").cast(pl.Float64, strict=False),
        pl.col("n").cast(pl.Float64, strict=False),
    ])
    out = out.filter(pl.col("n") > 0)
    out = _join_constants(out, constants, dataset, stat)

    if stat in PERCENT_MEAN_STATS:
        raw_prop = pl.col("raw") / 100.0
        reg_prop = (raw_prop * pl.col("n") + pl.col("K") * pl.col("mu")) / (
            pl.col("n") + pl.col("K")
        )
        out = out.with_columns(
            pl.when(
                pl.col("K").is_null() | pl.col("mu").is_null() | (pl.col("K") == 0)
            )
            .then(raw_prop)
            .otherwise(reg_prop)
            .alias("reg_prop")
        ).with_columns(
            [
                (raw_prop * 100.0).alias(f"{stat}_raw"),
                (pl.col("reg_prop") * 100.0).alias(f"{stat}_reg"),
                pl.col("n").alias(f"{stat}_n"),
            ]
        )
    else:
        reg = (pl.col("raw") * pl.col("n") + pl.col("K") * pl.col("mu")) / (
            pl.col("n") + pl.col("K")
        )
        out = out.with_columns(
            pl.when(
                pl.col("K").is_null() | pl.col("mu").is_null() | (pl.col("K") == 0)
            )
            .then(pl.col("raw"))
            .otherwise(reg)
            .alias(f"{stat}_reg")
        ).with_columns(
            [
                pl.col("raw").alias(f"{stat}_raw"),
                pl.col("n").alias(f"{stat}_n"),
            ]
        )
    return out.drop(["raw", "n", "mu", "K", "reg_prop"], strict=False)


def merge_frames(
    base: pl.DataFrame, frames: List[pl.DataFrame], keys: List[str]
) -> pl.DataFrame:
    for frame in frames:
        missing = [k for k in keys if k not in frame.columns]
        if missing:
            raise ValueError(
                f"Frame missing join keys {missing}. This would create row explosions."
            )
    # Process joins in batches to avoid deep lazy-plan stack overflows on
    # large DataFrames (polars ACCESS_VIOLATION with 30+ chained joins).
    merged = base
    batch_size = 8
    for i in range(0, len(frames), batch_size):
        batch = frames[i : i + batch_size]
        lazy = merged.lazy()
        for frame in batch:
            join_keys = [k for k in keys if k in frame.columns]
            # Deduplicate on join keys to prevent many-to-many row explosion
            frame_deduped = frame.unique(subset=join_keys, keep="last")
            lazy = lazy.join(frame_deduped.lazy(), on=join_keys, how="left")
        merged = lazy.collect()
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply regression using aggregate CSVs or parquets."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_REPO_DIR / "config" / "stability_config.yml",
        help="Path to stability config (yaml or json).",
    )
    parser.add_argument(
        "--constants",
        type=Path,
        default=_REPO_DIR / "config" / "stability_constants.csv",
        help="Path to stability_constants.csv.",
    )
    parser.add_argument(
        "--hitters",
        type=Path,
        default=_REPO_DIR / "data" / "output" / "damage_pos_2015_2025.parquet",
        help="Aggregated hitters CSV or parquet.",
    )
    parser.add_argument(
        "--pitchers",
        type=Path,
        default=_REPO_DIR / "data" / "output" / "pitcher_stuff_new.parquet",
        help="Aggregated pitchers CSV or parquet.",
    )
    parser.add_argument(
        "--pitch-types",
        type=Path,
        default=_REPO_DIR / "data" / "output" / "new_pitch_types.parquet",
        help="Aggregated pitch types CSV or parquet.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_DIR / "data" / "output",
        help="Output directory for regressed CSVs.",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=1,
        help="Decimal places for float output.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    stats_cfg = config.get("stats", {})
    constants = pl.read_csv(args.constants)

    outputs = {
        "hitters": (
            args.hitters,
            ["batter_mlbid", "hitter_name", "season", "level_id", "game_type_group"],
        ),
        "pitchers": (
            args.pitchers,
            ["pitcher_mlbid", "name", "season", "level_id", "pitcher_hand", "game_type_group"],
        ),
        "pitch_types": (
            args.pitch_types,
            [
                "pitcher_mlbid",
                "name",
                "pitcher_hand",
                "season",
                "level_id",
                "game_type_group",
                "pitch_tag",
            ],
        ),
    }

    for dataset, (path, keys) in outputs.items():
        if not path.exists():
            print(f"Missing {path}, skipping {dataset}.")
            continue
        if path.suffix.lower() == ".parquet":
            df = pl.read_parquet(path)
        else:
            df = pl.read_csv(path)
        frames: List[pl.DataFrame] = []

        for stat in stats_cfg.get(dataset, []):
            kind = stat.get("kind")
            name = stat.get("name")
            if kind == "derived":
                continue
            if kind == "rate":
                frame = apply_rate_from_agg(df, constants, dataset, name, keys)
            else:
                frame = apply_mean_from_agg(df, constants, dataset, name, keys)
            if frame is not None:
                frames.append(frame)

        # Derived: whiff_rate for contact_vs_avg if available
        if dataset == "hitters" and "whiff_rate_num" in df.columns:
            frame = apply_rate_from_agg(df, constants, dataset, "whiff_rate", keys)
            if frame is not None:
                frames.append(frame)

        effective_keys = [c for c in keys if c in df.columns]
        base = df.select(effective_keys).unique()
        merged = merge_frames(base, frames, effective_keys)

        if dataset == "hitters":
            merged = add_league_contact_baseline(merged, df)
            if (
                "selection_skill_raw" in merged.columns
                and "hittable_pitches_taken_raw" in merged.columns
            ):
                merged = merged.with_columns(
                    (
                        pl.col("selection_skill_raw")
                        - pl.col("hittable_pitches_taken_raw")
                    ).alias("SEAGER_raw")
                )
            if (
                "selection_skill_reg" in merged.columns
                and "hittable_pitches_taken_reg" in merged.columns
            ):
                merged = merged.with_columns(
                    (
                        pl.col("selection_skill_reg")
                        - pl.col("hittable_pitches_taken_reg")
                    ).alias("SEAGER_reg")
                )
            if (
                "pred_whiff_loc_mean_raw" in merged.columns
                and "whiff_rate_raw" in merged.columns
            ):
                if "lg_contact_baseline" in merged.columns:
                    merged = merged.with_columns(
                        (
                            pl.col("pred_whiff_loc_mean_raw")
                            - pl.col("whiff_rate_raw")
                            - pl.col("lg_contact_baseline")
                        ).alias("contact_vs_avg_raw")
                    )
                else:
                    merged = merged.with_columns(
                        (
                            pl.col("pred_whiff_loc_mean_raw")
                            - pl.col("whiff_rate_raw")
                        ).alias("contact_vs_avg_raw")
                    )
            if (
                "pred_whiff_loc_mean_raw" in merged.columns
                and "whiff_rate_reg" in merged.columns
            ):
                if (
                    "pred_whiff_loc_mean_reg" in merged.columns
                    and "lg_contact_baseline" in merged.columns
                ):
                    merged = merged.with_columns(
                        (
                            pl.col("pred_whiff_loc_mean_reg")
                            - pl.col("whiff_rate_reg")
                            - pl.col("lg_contact_baseline")
                        ).alias("contact_vs_avg_reg")
                    )
                elif "pred_whiff_loc_mean_reg" in merged.columns:
                    merged = merged.with_columns(
                        (
                            pl.col("pred_whiff_loc_mean_reg")
                            - pl.col("whiff_rate_reg")
                        ).alias("contact_vs_avg_reg")
                    )
            drop_cols = [c for c in merged.columns if c.startswith("whiff_rate_")]
            if drop_cols:
                merged = merged.drop(drop_cols)

        merged = round_floats(merged, args.round)
        out_path = args.out_dir / f"{dataset}_regressed.parquet"
        merged.write_parquet(out_path)
        print(f"Wrote {len(merged):,} rows to {out_path}")


if __name__ == "__main__":
    main()

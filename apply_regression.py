from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import polars as pl

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover
    yaml = None


ALIAS_FILTERS = {
    "swings": "swing == True",
    "bbe": "bbe == 1",
}

ALIAS_DENOM = {
    "pitches": "__len__",
    "swings": "swing == True",
    "bbe": "bbe == 1",
}

PERCENT_MEAN_STATS = {"pWhiff", "pred_whiff_pct", "pred_whiff_loc_mean"}


def load_config(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if yaml is None:  # pragma: no cover
        raise RuntimeError("PyYAML is required for .yml config files.")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _normalize_expr(expr: str) -> str:
    return expr.replace(" is in ", " in ").strip()


class ExprBuilder(ast.NodeVisitor):
    def visit_Name(self, node: ast.Name) -> pl.Expr:
        return pl.col(node.id)

    def visit_Constant(self, node: ast.Constant) -> pl.Expr:
        return pl.lit(node.value)

    def visit_BoolOp(self, node: ast.BoolOp) -> pl.Expr:
        values = [self.visit(v) for v in node.values]
        if not values:
            raise ValueError("Empty boolean expression.")
        expr = values[0]
        if isinstance(node.op, ast.And):
            for val in values[1:]:
                expr = expr & val
            return expr
        if isinstance(node.op, ast.Or):
            for val in values[1:]:
                expr = expr | val
            return expr
        raise ValueError(f"Unsupported boolean operator: {node.op}")

    def visit_Compare(self, node: ast.Compare) -> pl.Expr:
        if not node.ops:
            raise ValueError("Missing comparison operator.")

        exprs: List[pl.Expr] = []
        left = self.visit(node.left)
        for op, comp in zip(node.ops, node.comparators):
            if isinstance(op, ast.In):
                values = _literal_list(comp)
                exprs.append(left.is_in(values))
                right = None
            elif isinstance(op, ast.NotIn):
                values = _literal_list(comp)
                exprs.append(~left.is_in(values))
                right = None
            else:
                right = self.visit(comp)
                if isinstance(op, ast.Eq):
                    exprs.append(left == right)
                elif isinstance(op, ast.NotEq):
                    exprs.append(left != right)
                elif isinstance(op, ast.Gt):
                    exprs.append(left > right)
                elif isinstance(op, ast.GtE):
                    exprs.append(left >= right)
                elif isinstance(op, ast.Lt):
                    exprs.append(left < right)
                elif isinstance(op, ast.LtE):
                    exprs.append(left <= right)
                else:
                    raise ValueError(f"Unsupported comparator: {op}")
            if right is not None:
                left = right
        expr = exprs[0]
        for sub in exprs[1:]:
            expr = expr & sub
        return expr

    def visit_UnaryOp(self, node: ast.UnaryOp) -> pl.Expr:
        if isinstance(node.op, ast.Not):
            return ~self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -self.visit(node.operand)
        raise ValueError(f"Unsupported unary op: {node.op}")


def _literal_list(node: ast.AST) -> List[Any]:
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_literal(elt) for elt in node.elts]
    return [_literal(node)]


def _literal(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    raise ValueError("Expected a literal list or constant.")


def expr_from_string(expr: str) -> pl.Expr:
    expr = _normalize_expr(expr)
    tree = ast.parse(expr, mode="eval")
    return ExprBuilder().visit(tree.body)


def is_simple_name(token: str) -> bool:
    return token.isidentifier()


def count_expr_from_spec(spec: str) -> Tuple[Optional[pl.Expr], bool]:
    token = spec.strip()
    if token in ALIAS_DENOM:
        alias = ALIAS_DENOM[token]
        if alias == "__len__":
            return None, True
        return expr_from_string(alias), False
    if is_simple_name(token):
        return pl.col(token), False
    return expr_from_string(token), False


def filter_expr_for_stat(stat: Dict[str, Any]) -> Optional[pl.Expr]:
    if "filter" in stat and stat["filter"]:
        return expr_from_string(str(stat["filter"]))
    sample_n = stat.get("sample_n")
    if not sample_n:
        return None
    sample_n = str(sample_n).strip()
    if sample_n in ALIAS_FILTERS:
        return expr_from_string(ALIAS_FILTERS[sample_n])
    if is_simple_name(sample_n) and sample_n == "pitches":
        return None
    if is_simple_name(sample_n):
        return pl.col(sample_n)
    return expr_from_string(sample_n)


def tag_pitch(df: pl.DataFrame) -> pl.DataFrame:
    if "pitch_tag" in df.columns:
        return df
    return df.with_columns(
        pl.when(pl.col("pi_pitch_sub_type") == "SW")
        .then(pl.lit("SW"))
        .when(pl.col("pi_pitch_sub_type") == "SP")
        .then(pl.lit("SP"))
        .when(pl.col("pi_pitch_type") == "SI")
        .then(pl.lit("SI"))
        .when((pl.col("pi_pitch_group") == "FA") & (pl.col("pi_pitch_type") == "FC"))
        .then(pl.lit("HC"))
        .when(pl.col("pi_pitch_type") == "FS")
        .then(pl.lit("FS"))
        .when(pl.col("pi_pitch_type") == "FA")
        .then(pl.lit("FA"))
        .when(pl.col("pi_pitch_group") == "SL")
        .then(pl.lit("SL"))
        .when(pl.col("pi_pitch_type") == "CH")
        .then(pl.lit("CH"))
        .when(pl.col("pi_pitch_group") == "CU")
        .then(pl.lit("CU"))
        .otherwise(pl.lit("XX"))
        .alias("pitch_tag")
    )


def round_floats(df: pl.DataFrame, places: int = 3) -> pl.DataFrame:
    float_cols = [
        c
        for c, d in zip(df.columns, df.dtypes)
        if d in (pl.Float32, pl.Float64)
    ]
    if not float_cols:
        return df
    return df.with_columns([pl.col(c).round(places) for c in float_cols])


def normalize_long_schema(df: pl.DataFrame) -> pl.DataFrame:
    int_cols = [
        c
        for c, d in zip(df.columns, df.dtypes)
        if d in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
    ]
    float_cols = [
        c
        for c, d in zip(df.columns, df.dtypes)
        if d in (pl.Float32, pl.Float64)
    ]
    exprs = []
    if int_cols:
        exprs.extend([pl.col(c).cast(pl.Int64) for c in int_cols])
    if float_cols:
        exprs.extend([pl.col(c).cast(pl.Float64) for c in float_cols])
    return df.with_columns(exprs) if exprs else df


def apply_rate_stat(
    df: pl.DataFrame,
    stat: Dict[str, Any],
    group_cols: List[str],
    constants: pl.DataFrame,
    dataset: str,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    num_expr, num_len = count_expr_from_spec(str(stat["numerator"]))
    den_expr, den_len = count_expr_from_spec(str(stat["denominator"]))

    agg_exprs = []
    if num_len:
        agg_exprs.append(pl.len().alias("num"))
    else:
        agg_exprs.append(num_expr.cast(pl.Int64).sum().alias("num"))
    if den_len:
        agg_exprs.append(pl.len().alias("den"))
    else:
        agg_exprs.append(den_expr.cast(pl.Int64).sum().alias("den"))

    player = (
        df.group_by(group_cols)
        .agg(agg_exprs)
        .with_columns((pl.col("num") / pl.col("den")).alias("raw"))
        .filter(pl.col("den") > 0)
    )

    const = constants.filter(
        (pl.col("dataset") == dataset) & (pl.col("stat") == stat["name"])
    )
    join_keys = [
        c
        for c in ["level_id", "pitch_tag", "season"]
        if c in const.columns and c in player.columns
    ]
    if join_keys:
        player = player.join(const, on=join_keys, how="left")
    else:
        player = player.join(const, how="cross")

    reg_expr = (pl.col("num") + pl.col("K") * pl.col("mu")) / (
        pl.col("den") + pl.col("K")
    )
    player = player.with_columns(
        pl.when(
            pl.col("K").is_null()
            | pl.col("mu").is_null()
            | (pl.col("K") == 0)
        )
        .then(pl.col("raw"))
        .otherwise(reg_expr)
        .alias("reg")
    )

    player = player.with_columns(
        [
            (pl.col("raw") * 100.0).alias("raw"),
            (pl.col("reg") * 100.0).alias("reg"),
        ]
    )

    wide = player.select(
        group_cols
        + [
            pl.col("num").alias(f"{stat['name']}_num"),
            pl.col("den").alias(f"{stat['name']}_n"),
            pl.col("raw").alias(f"{stat['name']}_raw"),
            pl.col("reg").alias(f"{stat['name']}_reg"),
        ]
    )

    long = player.select(
        group_cols
        + [
            pl.lit(dataset).alias("dataset"),
            pl.lit(stat["name"]).alias("stat"),
            pl.col("num"),
            pl.col("den").alias("n"),
            pl.col("raw"),
            pl.col("reg"),
            pl.col("mu"),
            pl.col("K"),
        ]
    )
    long = normalize_long_schema(long)
    return wide, long


def apply_value_stat(
    df: pl.DataFrame,
    stat: Dict[str, Any],
    group_cols: List[str],
    constants: pl.DataFrame,
    dataset: str,
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    value = stat.get("value")
    if value is None:
        raise ValueError(f"Missing value for stat {stat['name']}")
    value_expr = pl.col(str(value))

    filter_expr = filter_expr_for_stat(stat)
    if filter_expr is not None:
        value_expr = pl.when(filter_expr).then(value_expr)

    kind = stat["kind"]
    if kind == "quantile":
        q = float(stat.get("quantile", 0.5))
        stat_expr = value_expr.quantile(q)
    elif kind == "max":
        stat_expr = value_expr.max()
    else:
        stat_expr = value_expr.mean()

    player = df.group_by(group_cols).agg(
        [
            stat_expr.alias("raw"),
            value_expr.count().alias("n"),
        ]
    ).filter(pl.col("n") > 0)

    const = constants.filter(
        (pl.col("dataset") == dataset) & (pl.col("stat") == stat["name"])
    )
    join_keys = [
        c
        for c in ["level_id", "pitch_tag", "season"]
        if c in const.columns and c in player.columns
    ]
    if join_keys:
        player = player.join(const, on=join_keys, how="left")
    else:
        player = player.join(const, how="cross")

    reg_expr = (pl.col("raw") * pl.col("n") + pl.col("K") * pl.col("mu")) / (
        pl.col("n") + pl.col("K")
    )
    player = player.with_columns(
        pl.when(
            pl.col("K").is_null()
            | pl.col("mu").is_null()
            | (pl.col("K") == 0)
        )
        .then(pl.col("raw"))
        .otherwise(reg_expr)
        .alias("reg")
    )

    if stat["name"] in PERCENT_MEAN_STATS:
        player = player.with_columns(
            [
                (pl.col("raw") * 100.0).alias("raw"),
                (pl.col("reg") * 100.0).alias("reg"),
            ]
        )

    wide = player.select(
        group_cols
        + [
            pl.col("n").alias(f"{stat['name']}_n"),
            pl.col("raw").alias(f"{stat['name']}_raw"),
            pl.col("reg").alias(f"{stat['name']}_reg"),
        ]
    )

    long = player.select(
        group_cols
        + [
            pl.lit(dataset).alias("dataset"),
            pl.lit(stat["name"]).alias("stat"),
            pl.col("n"),
            pl.col("raw"),
            pl.col("reg"),
            pl.col("mu"),
            pl.col("K"),
        ]
    )
    long = normalize_long_schema(long)
    return wide, long


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply regression constants.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("stability_config.yml"),
        help="Path to stability config (yaml or json).",
    )
    parser.add_argument(
        "--parquet-path",
        type=Path,
        required=True,
        help="Path to pitch-level parquet file.",
    )
    parser.add_argument(
        "--constants",
        type=Path,
        default=Path("stability_constants.csv"),
        help="Path to stability_constants.csv.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="Output directory for CSVs.",
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

    pitch = pl.read_parquet(args.parquet_path)
    pitch = tag_pitch(pitch)

    constants = pl.read_csv(args.constants)

    datasets = {
        "hitters": {
            "group_cols": ["batter_mlbid", "season", "level_id"],
            "meta_cols": ["hitter_name"],
        },
        "pitchers": {
            "group_cols": ["pitcher_mlbid", "season", "level_id"],
            "meta_cols": ["name", "pitcher_hand"],
        },
        "pitch_types": {
            "group_cols": ["pitcher_mlbid", "season", "level_id", "pitch_tag"],
            "meta_cols": ["name", "pitcher_hand"],
        },
    }

    long_frames: List[pl.DataFrame] = []

    for dataset, cfg in datasets.items():
        group_cols = cfg["group_cols"]
        meta_cols = cfg["meta_cols"]

        base = pitch.group_by(group_cols).agg(
            [pl.first(c).alias(c) for c in meta_cols if c in pitch.columns]
        )

        wide_frames: List[pl.DataFrame] = [base]

        for stat in stats_cfg.get(dataset, []):
            kind = stat.get("kind")
            name = stat.get("name")
            if kind == "derived":
                continue
            if kind == "rate":
                wide, long = apply_rate_stat(
                    pitch, stat, group_cols, constants, dataset
                )
            else:
                wide, long = apply_value_stat(
                    pitch, stat, group_cols, constants, dataset
                )
            wide_frames.append(wide)
            long_frames.append(long)

        merged = wide_frames[0]
        for frame in wide_frames[1:]:
            merged = merged.join(frame, on=group_cols, how="left")

        # Derived stats (raw/reg) that depend on regressed components
        if dataset == "hitters":
            whiff_stat = {
                "name": "whiff_rate",
                "kind": "rate",
                "numerator": "whiff == True",
                "denominator": "swing == True",
            }
            whiff_wide, _ = apply_rate_stat(
                pitch, whiff_stat, group_cols, constants, dataset
            )
            merged = merged.join(whiff_wide, on=group_cols, how="left")

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
                merged = merged.with_columns(
                    (
                        pl.col("pred_whiff_loc_mean_raw")
                        - pl.col("whiff_rate_raw")
                    ).alias("contact_vs_avg_raw")
                )
            if (
                "pred_whiff_loc_mean_reg" in merged.columns
                and "whiff_rate_reg" in merged.columns
            ):
                merged = merged.with_columns(
                    (
                        pl.col("pred_whiff_loc_mean_reg")
                        - pl.col("whiff_rate_reg")
                    ).alias("contact_vs_avg_reg")
                )
            drop_cols = [
                c
                for c in merged.columns
                if c.startswith("whiff_rate_")
            ]
            if drop_cols:
                merged = merged.drop(drop_cols)

        merged = round_floats(merged, args.round)
        out_path = args.out_dir / f"{dataset}_regressed.csv"
        merged.write_csv(out_path)
        print(f"Wrote {len(merged):,} rows to {out_path}")

    if long_frames:
        long_df = pl.concat(long_frames, how="diagonal")
        long_df = round_floats(long_df, args.round)
        out_path = args.out_dir / "regression_long.csv"
        long_df.write_csv(out_path)
        print(f"Wrote {len(long_df):,} rows to {out_path}")


if __name__ == "__main__":
    main()

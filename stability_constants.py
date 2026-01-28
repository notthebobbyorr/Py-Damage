from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
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


def grouped_rate(
    df: pl.DataFrame,
    stat: Dict[str, Any],
    group_cols: List[str],
    min_denom: int,
) -> pl.DataFrame:
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

    return (
        df.group_by(group_cols)
        .agg(agg_exprs)
        .with_columns((pl.col("num") / pl.col("den")).alias("stat"))
        .filter(pl.col("den") >= min_denom)
    )


def grouped_value(
    df: pl.DataFrame,
    stat: Dict[str, Any],
    group_cols: List[str],
    min_denom: int,
) -> pl.DataFrame:
    filter_expr = filter_expr_for_stat(stat)
    value = stat.get("value")
    if value is None:
        raise ValueError(f"Missing value for stat: {stat['name']}")
    value_expr = pl.col(str(value))
    if filter_expr is not None:
        value_expr = pl.when(filter_expr).then(value_expr)

    agg = [
        value_expr.mean().alias("stat"),
        value_expr.var(ddof=1).alias("var"),
        value_expr.count().alias("n"),
    ]

    if stat["kind"] == "quantile":
        q = float(stat.get("quantile", 0.5))
        agg[0] = value_expr.quantile(q).alias("stat")
    elif stat["kind"] == "max":
        agg[0] = value_expr.max().alias("stat")

    return (
        df.group_by(group_cols)
        .agg(agg)
        .filter(pl.col("n") >= min_denom)
    )


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


def rate_constants(
    player: pl.DataFrame,
    context_cols: List[str],
    min_players: int,
) -> pl.DataFrame:
    group_mu = player.group_by(context_cols).agg(
        [
            pl.mean("stat").alias("mu"),
            pl.var("stat").alias("var_obs"),
            pl.len().alias("n_players"),
        ]
    )
    player_mu = player.join(group_mu.select(context_cols + ["mu"]), on=context_cols)
    player_mu = player_mu.with_columns(
        (pl.col("mu") * (1 - pl.col("mu")) / pl.col("den")).alias("var_within_i")
    )
    var_within = player_mu.group_by(context_cols).agg(
        pl.mean("var_within_i").alias("var_within")
    )
    stats = group_mu.join(var_within, on=context_cols, how="left")
    stats = stats.with_columns(
        pl.when(pl.col("var_obs").is_not_null() & pl.col("var_within").is_not_null())
        .then(pl.col("var_obs") - pl.col("var_within"))
        .otherwise(None)
        .alias("var_true_raw")
    ).with_columns(
        pl.when(pl.col("var_true_raw").is_not_null())
        .then(pl.max_horizontal(pl.lit(1e-8), pl.col("var_true_raw")))
        .otherwise(None)
        .alias("var_true")
    )

    k_expr = (pl.col("mu") * (1 - pl.col("mu")) / pl.col("var_true")) - 1
    stats = stats.with_columns(
        pl.when(pl.col("var_true").is_not_null())
        .then(pl.max_horizontal(pl.lit(0.0), k_expr))
        .otherwise(None)
        .alias("K")
    )
    stats = stats.with_columns(
        [
            (pl.col("K") * 0.5 / 0.5).alias("n_50"),
            (pl.col("K") * 0.7 / 0.3).alias("n_70"),
            (pl.col("K") * 0.8 / 0.2).alias("n_80"),
        ]
    )
    return stats.filter(pl.col("n_players") >= min_players)


def mean_constants(
    player: pl.DataFrame,
    context_cols: List[str],
    min_players: int,
) -> pl.DataFrame:
    group_stats = player.group_by(context_cols).agg(
        [
            pl.mean("stat").alias("mu"),
            pl.var("stat").alias("var_obs"),
            pl.len().alias("n_players"),
            (1 / pl.col("n")).mean().alias("mean_inv_n"),
        ]
    )
    within_stats = (
        player.filter(pl.col("n") > 1)
        .group_by(context_cols)
        .agg(
            [
                (pl.col("var") * (pl.col("n") - 1)).sum().alias("within_num"),
                (pl.col("n") - 1).sum().alias("within_den"),
            ]
        )
        .with_columns((pl.col("within_num") / pl.col("within_den")).alias("var_within"))
        .drop(["within_num", "within_den"])
    )
    stats = group_stats.join(within_stats, on=context_cols, how="left")
    stats = stats.with_columns(
        pl.when(pl.col("var_within").is_not_null() & pl.col("var_obs").is_not_null())
        .then(pl.col("var_obs") - pl.col("var_within") * pl.col("mean_inv_n"))
        .otherwise(None)
        .alias("var_between_raw")
    ).with_columns(
        pl.when(pl.col("var_between_raw").is_not_null())
        .then(pl.max_horizontal(pl.lit(1e-8), pl.col("var_between_raw")))
        .otherwise(None)
        .alias("var_between")
    )
    k_expr = pl.col("var_within") / pl.col("var_between")
    stats = stats.with_columns(
        pl.when(pl.col("var_between").is_not_null())
        .then(pl.max_horizontal(pl.lit(0.0), k_expr))
        .otherwise(None)
        .alias("K")
    )
    stats = stats.with_columns(
        [
            (pl.col("K") * 0.5 / 0.5).alias("n_50"),
            (pl.col("K") * 0.7 / 0.3).alias("n_70"),
            (pl.col("K") * 0.8 / 0.2).alias("n_80"),
        ]
    )
    return stats.filter(pl.col("n_players") >= min_players)


def bootstrap_constants(
    player: pl.DataFrame,
    context_cols: List[str],
    min_players: int,
    kind: str,
    q: float,
    iters: int,
) -> pl.DataFrame:
    rows: List[Dict[str, Any]] = []
    rng = np.random.default_rng(7)
    for row in player.iter_rows(named=True):
        values = row.get("values")
        n = row.get("n")
        if values is None or n is None or n < 2:
            continue
        values_arr = np.array(values, dtype=float)
        if values_arr.size < 2:
            continue
        stats = []
        for _ in range(iters):
            sample = rng.choice(values_arr, size=values_arr.size, replace=True)
            if kind == "quantile":
                stats.append(float(np.quantile(sample, q)))
            else:
                stats.append(float(np.max(sample)))
        var_within = float(np.var(stats, ddof=1)) if len(stats) > 1 else 0.0
        rows.append(
            {
                **{col: row[col] for col in context_cols},
                "stat": row["stat"],
                "n": n,
                "var_within": var_within,
            }
        )
    if not rows:
        return pl.DataFrame()
    boot = pl.DataFrame(rows)
    group = boot.group_by(context_cols).agg(
        [
            pl.mean("stat").alias("mu"),
            pl.var("stat").alias("var_obs"),
            pl.mean("var_within").alias("var_within"),
            pl.len().alias("n_players"),
        ]
    )
    stats = group.with_columns(
        pl.when(pl.col("var_obs").is_not_null() & pl.col("var_within").is_not_null())
        .then(pl.max_horizontal(pl.lit(1e-8), pl.col("var_obs") - pl.col("var_within")))
        .otherwise(None)
        .alias("var_between")
    )
    boot = boot.join(stats.select(context_cols + ["var_between"]), on=context_cols)
    boot = boot.with_columns(
        (pl.col("var_between") / (pl.col("var_between") + pl.col("var_within")))
        .alias("r_i")
    )
    boot = boot.with_columns(
        (pl.col("n") * (1 - pl.col("r_i")) / pl.col("r_i")).alias("K_i")
    )
    k_stats = (
        boot.group_by(context_cols)
        .agg(pl.median("K_i").alias("K"))
        .join(stats, on=context_cols, how="left")
    )
    k_stats = k_stats.with_columns(
        [
            (pl.col("K") * 0.5 / 0.5).alias("n_50"),
            (pl.col("K") * 0.7 / 0.3).alias("n_70"),
            (pl.col("K") * 0.8 / 0.2).alias("n_80"),
        ]
    )
    return k_stats.filter(pl.col("n_players") >= min_players)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute stability constants.")
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
        "--out",
        type=Path,
        default=Path("stability_constants.csv"),
        help="Output CSV for constants.",
    )
    parser.add_argument(
        "--min-players",
        type=int,
        default=10,
        help="Minimum players per group to emit constants.",
    )
    parser.add_argument(
        "--min-denom",
        type=int,
        default=20,
        help="Minimum denominator/sample_n required for player-level stats.",
    )
    parser.add_argument(
        "--by-season",
        action="store_true",
        help="Emit constants split by season (default: pooled across seasons).",
    )
    parser.add_argument(
        "--bootstrap-iters",
        type=int,
        default=0,
        help="Bootstrap iterations for quantile/max (0 skips).",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    stats_cfg = config.get("stats", {})

    pitch = pl.read_parquet(args.parquet_path)
    pitch = tag_pitch(pitch)

    if args.by_season:
        hitter_context = ["season", "level_id"]
        pitcher_context = ["season", "level_id"]
        pitch_type_context = ["season", "level_id", "pitch_tag"]
    else:
        hitter_context = ["level_id"]
        pitcher_context = ["level_id"]
        pitch_type_context = ["level_id", "pitch_tag"]

    datasets = {
        "hitters": {
            "group_cols": ["batter_mlbid", "season", "level_id"],
            "context_cols": hitter_context,
        },
        "pitchers": {
            "group_cols": ["pitcher_mlbid", "season", "level_id"],
            "context_cols": pitcher_context,
        },
        "pitch_types": {
            "group_cols": ["pitcher_mlbid", "season", "level_id", "pitch_tag"],
            "context_cols": pitch_type_context,
        },
    }

    outputs: List[pl.DataFrame] = []

    for dataset, cfg in datasets.items():
        group_cols = cfg["group_cols"]
        context_cols = cfg["context_cols"]
        for stat in stats_cfg.get(dataset, []):
            kind = stat.get("kind")
            name = stat.get("name")
            if kind == "derived":
                continue
            if kind == "rate":
                player = grouped_rate(pitch, stat, group_cols, args.min_denom)
                stats = rate_constants(player, context_cols, args.min_players)
            elif kind in {"mean", "quantile", "max"}:
                player = grouped_value(pitch, stat, group_cols, args.min_denom)
                if kind in {"quantile", "max"}:
                    if args.bootstrap_iters <= 0:
                        continue
                    value = stat.get("value")
                    filter_expr = filter_expr_for_stat(stat)
                    value_expr = pl.col(str(value))
                    if filter_expr is not None:
                        value_expr = pl.when(filter_expr).then(value_expr)
                    q = float(stat.get("quantile", 0.5))
                    player = pitch.group_by(group_cols).agg(
                        [
                            value_expr.drop_nulls().alias("values"),
                            value_expr.count().alias("n"),
                            (
                                value_expr.quantile(q)
                                if kind == "quantile"
                                else value_expr.max()
                            ).alias("stat"),
                        ]
                    )
                    player = player.filter(pl.col("n") >= args.min_denom)
                    stats = bootstrap_constants(
                        player,
                        context_cols,
                        args.min_players,
                        kind,
                        q,
                        args.bootstrap_iters,
                    )
                else:
                    stats = mean_constants(player, context_cols, args.min_players)
            else:
                continue

            if stats.is_empty():
                continue

            stats = stats.with_columns(
                [
                    pl.lit(dataset).alias("dataset"),
                    pl.lit(name).alias("stat"),
                    pl.lit(kind).alias("kind"),
                ]
            )
            outputs.append(stats)

    if outputs:
        result = pl.concat(outputs, how="diagonal")
        desired_cols = [
            "dataset",
            "stat",
            "kind",
            "season",
            "level_id",
            "pitch_tag",
            "n_players",
            "mu",
            "var_obs",
            "var_within",
            "var_true",
            "var_between",
            "K",
            "n_50",
            "n_70",
            "n_80",
        ]
        existing = [c for c in desired_cols if c in result.columns]
        result = result.select(existing)
        result.write_csv(args.out)
        print(f"Wrote {len(result):,} rows to {args.out}")
    else:
        print("No stability constants produced.")


if __name__ == "__main__":
    main()

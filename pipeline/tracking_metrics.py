"""Observed miss-distance means and sample sizes, in source inches."""
import polars as pl


def miss_distance_expressions(df):
    value = (pl.col('miss_distance').cast(pl.Float64, strict=False).fill_nan(None)
             if 'miss_distance' in df.columns else pl.lit(None, dtype=pl.Float64))
    return [value.mean().alias('miss_distance'), value.count().alias('miss_distance_n')]

"""Pooled observed swing-rate counts and percentages for generated statistics."""
import polars as pl


def swing_count_expressions(df):
    return [(pl.col('whiff') == 1).sum().alias('rate_whiffs'),
            (pl.col('swing') == 1).sum().alias('rate_swings'),
            pl.len().alias('rate_pitches')]


def add_swing_rates(frame):
    if frame.is_empty():
        return frame
    columns = frame.columns

    def first(names):
        choices = [pl.col(c).cast(pl.Float64, strict=False) for c in names if c in columns]
        return pl.coalesce(choices) if choices else pl.lit(None, dtype=pl.Float64)

    whiffs = first(['rate_whiffs', 'whiff_rate_num', 'SwStr_num', 'whiffs'])
    swings = first(['rate_swings', 'whiff_rate_den', 'Swing_pct_num', 'swings'])
    pitches = first(['rate_pitches', 'pitches', 'Swing_pct_den', 'SwStr_den'])
    swing = pl.coalesce(pl.when(pitches > 0).then(100 * swings / pitches), first(['Swing_pct']))
    swstr = pl.coalesce(pl.when(pitches > 0).then(100 * whiffs / pitches), first(['SwStr']))
    whiff = pl.coalesce(pl.when(swings > 0).then(100 * whiffs / swings), first(['whiff_pct']),
                        pl.when(swing > 0).then(100 * swstr / swing))
    return frame.with_columns(swing.alias('Swing_pct'), whiff.alias('whiff_pct'),
                              pl.coalesce(swstr, whiff * swing / 100).alias('SwStr'))

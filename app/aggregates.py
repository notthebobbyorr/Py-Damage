from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import streamlit as st

from app.filters import (
    download_button,
    filter_by_game_type_group,
    filter_by_team_token,
    filter_by_values,
    game_type_group_options,
    player_id_options,
    season_options,
    team_options,
)
from app.viz import render_table


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SpanSpec:
    """Describes how to aggregate one gamelog kind over a date range.

    rate_specs:   (display_label, numerator_col, denominator_col) -> 100*num/den
    wmean_specs:  (display_label, value_col, weight_col) -> sum(value*weight)/sum(weight)
    context_cols: raw count columns kept alongside rates for context (e.g. PA, BBE)
    team_col:     team-code column to collapse into a joined string when it is NOT
                  a group key (player specs). For team specs the team IS the group
                  key, so leave this None.
    """

    group_keys: list[str]
    count_cols: list[str]
    rate_specs: list[tuple[str, str, str]] = field(default_factory=list)
    wmean_specs: list[tuple[str, str, str]] = field(default_factory=list)
    context_cols: list[str] = field(default_factory=list)
    team_col: str | None = None
    name_col: str | None = None


_HITTER_RATES = [
    ("Damage/BBE (%)", "damaged_bbe", "bbe"),
    ("Pulled FB (%)", "pulled_fbs", "bbe"),
    ("LA>=20%", "la_gte_20_bbe", "bbe"),
    ("LA<=0%", "la_lte_0_bbe", "bbe"),
    ("K%", "K", "PA"),
    ("BB%", "BB", "PA"),
    ("HR/PA (%)", "HR", "PA"),
    ("Whiff%", "whiffs", "swings"),
]
_HITTER_COUNTS = [
    "PA", "bbe", "HR", "XBH", "hits", "damaged_bbe", "pulled_fbs",
    "la_gte_20_bbe", "la_lte_0_bbe", "BB", "K", "pitches", "FA", "BR", "OFF",
    "swings", "chases", "whiffs", "selective_takes", "hittable_takes",
    "vs_RHP", "vs_LHP",
]

_PITCHER_RATES = [
    ("SwStr (%)", "whiffs", "pitches"),
    ("Swing (%)", "swings", "pitches"),
    ("Zone (%)", "zone_pitches", "pitches"),
    ("Ball (%)", "balls", "pitches"),
    ("Chase (%)", "chases", "out_of_zone"),
    ("FA Usage (%)", "FA", "pitches"),
    ("Damage/BBE (%)", "damaged_bbe", "bbe"),
    ("LA<=0%", "la_lte_0_bbe", "bbe"),
    ("LA>=20%", "la_gte_20_bbe", "bbe"),
    ("K%", "K", "TBF"),
    ("BB%", "BB", "TBF"),
]
_PITCHER_COUNTS = [
    "TBF", "pitches", "bbe", "damaged_bbe", "HR", "XBH", "hits",
    "la_gte_20_bbe", "la_lte_0_bbe", "BB", "K", "swings", "chases", "whiffs",
    "zone_pitches", "out_of_zone", "strikes", "balls", "FA", "BR", "OFF",
    "vs_LHB", "vs_RHB",
]

_PITCH_TYPE_RATES = [
    ("SwStr (%)", "whiffs", "pitches"),
    ("Swing (%)", "swings", "pitches"),
    ("Zone (%)", "zone_pitches", "pitches"),
    ("Ball (%)", "balls", "pitches"),
    ("Chase (%)", "chases", "out_of_zone"),
    ("Damage/BBE (%)", "damaged_bbe", "bbe"),
    ("LA<=0%", "la_lte_0_bbe", "bbe"),
    ("LA>=20%", "la_gte_20_bbe", "bbe"),
]
_PITCH_TYPE_COUNTS = [
    "pitches", "bbe", "damaged_bbe", "HR", "XBH", "hits",
    "la_lte_0_bbe", "la_gte_20_bbe", "swings", "chases", "whiffs", "BB", "K",
    "zone_pitches", "strikes", "balls", "out_of_zone", "vs_LHB", "vs_RHB",
]


HITTER_SPEC = SpanSpec(
    group_keys=["batter_mlbid"],
    count_cols=_HITTER_COUNTS,
    rate_specs=_HITTER_RATES,
    context_cols=["PA", "bbe"],
    team_col="hitting_code",
    name_col="hitter_name",
)
PITCHER_SPEC = SpanSpec(
    group_keys=["pitcher_mlbid"],
    count_cols=_PITCHER_COUNTS,
    rate_specs=_PITCHER_RATES,
    wmean_specs=[
        ("Avg FA mph", "FA_mph", "FA"),
        ("Pitch Grade", "stuff", "pitches"),
        ("Exec Grade", "grade_v13", "pitches"),
    ],
    context_cols=["TBF", "pitches", "bbe"],
    team_col="pitching_code",
    name_col="pitcher_name",
)
PITCH_TYPE_SPEC = SpanSpec(
    group_keys=["pitcher_mlbid", "pitch_tag"],
    count_cols=_PITCH_TYPE_COUNTS,
    rate_specs=_PITCH_TYPE_RATES,
    wmean_specs=[
        ("Avg mph", "velo", "pitches"),
        ("Pitch Grade", "stuff", "pitches"),
        ("Exec Grade", "grade_v13", "pitches"),
    ],
    context_cols=["pitches", "bbe"],
    team_col="pitching_code",
    name_col="pitcher_name",
)
TEAM_HITTER_SPEC = SpanSpec(
    group_keys=["hitting_code"],
    count_cols=_HITTER_COUNTS,
    rate_specs=_HITTER_RATES,
    context_cols=["PA", "bbe"],
)
TEAM_PITCHER_SPEC = SpanSpec(
    group_keys=["pitching_code"],
    count_cols=_PITCHER_COUNTS,
    rate_specs=_PITCHER_RATES,
    wmean_specs=[
        ("Pitch Grade", "stuff", "pitches"),
        ("Exec Grade", "grade_v13", "pitches"),
    ],
    context_cols=["TBF", "pitches", "bbe"],
)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _add_rates(df: pd.DataFrame, spec: SpanSpec) -> pd.DataFrame:
    """Compute rate columns row-wise from num/den count columns already present."""
    out = df.copy()
    for label, num, den in spec.rate_specs:
        if num in out.columns and den in out.columns:
            n = pd.to_numeric(out[num], errors="coerce")
            d = pd.to_numeric(out[den], errors="coerce")
            out[label] = np.where(d > 0, 100.0 * n / d, np.nan)
    return out


def _clean_team_options(df: pd.DataFrame, col: str, include_all: bool) -> list[str]:
    """Team dropdown options with unresolved numeric team codes removed.

    At MLB level the source data carries some teams as raw numeric StatsAPI IDs
    (e.g. '121') in addition to the abbreviation ('NYM'); the abbreviation already
    holds the full team season, so the numeric duplicates are dropped from the
    picker. Mirrors the '^\\d+$' convention used elsewhere for team codes."""
    opts = [
        o
        for o in team_options(df, col)
        if o == "All" or not re.fullmatch(r"\d+", str(o).strip())
    ]
    if not include_all:
        opts = [o for o in opts if o != "All"]
    return opts


def _join_team_codes(s: pd.Series) -> str:
    """Join distinct non-numeric team codes, matching the Individual Stats convention.

    Mirrors build_hitters/build_pitchers: unresolved numeric team IDs are dropped so
    a player who changed teams shows e.g. 'ARI | BOS' rather than fanning into rows."""
    vals = {
        str(v).strip()
        for v in pd.unique(s.dropna())
        if str(v).strip() and not re.fullmatch(r"\d+", str(v).strip())
    }
    return " | ".join(sorted(vals))


def aggregate_span(df_range: pd.DataFrame, spec: SpanSpec, mode: str) -> pd.DataFrame:
    """Collapse the in-range rows into one summary row per group."""
    keys = [c for c in spec.group_keys if c in df_range.columns]
    if not keys:
        return pd.DataFrame()
    counts = [c for c in spec.count_cols if c in df_range.columns]
    work = df_range.copy()
    for c in counts:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    grouped = work.groupby(keys, observed=True, dropna=False)
    agg = grouped[counts].sum(min_count=1).reset_index()

    if "game_pk" in work.columns:
        games = grouped["game_pk"].nunique().reset_index(name="G")
        agg = agg.merge(games, on=keys, how="left")

    # When grouping by player ID, carry a representative name (most frequent
    # spelling) so minor name variants on the same ID don't split into rows.
    if spec.name_col and spec.name_col in work.columns and spec.name_col not in keys:
        name_df = (
            work.groupby(keys + [spec.name_col], observed=True, dropna=False)
            .size()
            .reset_index(name="__n")
            .sort_values("__n")
            .drop_duplicates(subset=keys, keep="last")[keys + [spec.name_col]]
        )
        agg = agg.merge(name_df, on=keys, how="left")

    # When grouping by player, collapse the team code into a joined string so a
    # traded / multi-level player produces one row, not one per team code.
    if spec.team_col and spec.team_col in work.columns and spec.team_col not in keys:
        teams = (
            grouped[spec.team_col]
            .agg(_join_team_codes)
            .reset_index()
        )
        agg = agg.merge(teams, on=keys, how="left")

    if mode == "Rates":
        agg = _add_rates(agg, spec)
        for label, val, wt in spec.wmean_specs:
            if val in work.columns and wt in work.columns:
                v = pd.to_numeric(work[val], errors="coerce")
                w = pd.to_numeric(work[wt], errors="coerce")
                tmp = pd.DataFrame(
                    {
                        "__wv": v * w,
                        "__w": w.where(v.notna(), 0.0),
                    }
                )
                tmp[keys] = work[keys].values
                sums = tmp.groupby(keys, observed=True, dropna=False)[
                    ["__wv", "__w"]
                ].sum(min_count=1).reset_index()
                sums[label] = np.where(
                    sums["__w"] > 0, sums["__wv"] / sums["__w"], np.nan
                )
                agg = agg.merge(sums[keys + [label]], on=keys, how="left")
    return agg


# ---------------------------------------------------------------------------
# Display assembly
# ---------------------------------------------------------------------------
def _display_columns(spec: SpanSpec, mode: str) -> list[str]:
    """Ordered raw column names to show (pre-rename), excluding the label col."""
    # Lead columns: representative name, joined team, then any remaining
    # non-ID group keys (e.g. pitch_tag).
    id_cols: list[str] = []
    if spec.name_col:
        id_cols.append(spec.name_col)
    if spec.team_col and spec.team_col not in id_cols:
        id_cols.append(spec.team_col)
    for col in spec.group_keys:
        if col.endswith("_mlbid") or col == spec.name_col or col in id_cols:
            continue
        id_cols.append(col)
    if mode == "Raw counts":
        return id_cols + ["G"] + spec.count_cols
    rate_labels = [lbl for lbl, _, _ in spec.rate_specs]
    wmean_labels = [lbl for lbl, _, _ in spec.wmean_specs]
    ctx = [c for c in spec.context_cols]
    return id_cols + ["G"] + ctx + wmean_labels + rate_labels


def build_span_table(
    df_range: pd.DataFrame,
    spec: SpanSpec,
    mode: str,
    label_col: str,
    span_label: str,
) -> pd.DataFrame:
    """Aggregate row(s) on top, per-game rows beneath, sharing one layout."""
    if df_range.empty:
        return pd.DataFrame()

    agg = aggregate_span(df_range, spec, mode)
    games = df_range.copy()
    if mode == "Rates":
        games = _add_rates(games, spec)
        for label, val, _wt in spec.wmean_specs:
            if val in games.columns:
                games[label] = pd.to_numeric(games[val], errors="coerce")

    cols = [c for c in _display_columns(spec, mode) if c in agg.columns or c in games.columns]

    agg = agg.reindex(columns=cols)
    agg.insert(0, label_col, span_label)

    if "game_date" in games.columns:
        games = games.sort_values("game_date", ascending=False)
        date_disp = pd.to_datetime(games["game_date"], errors="coerce").dt.strftime("%m/%d/%Y")
    else:
        date_disp = pd.Series([""] * len(games), index=games.index)
    games_view = games.reindex(columns=cols)
    games_view.insert(0, label_col, date_disp.values)

    return pd.concat([agg, games_view], ignore_index=True)


# ---------------------------------------------------------------------------
# Tab UI
# ---------------------------------------------------------------------------
def render_span_tab(
    full_df: pd.DataFrame,
    spec: SpanSpec,
    *,
    level_map: dict[str, list[int]],
    key_prefix: str,
    entity: str,  # "player" or "team"
    team_col: str,
    rename_map: dict[str, str],
    id_col: str | None = None,
    name_col: str | None = None,
    pitch_tag_filter: bool = False,
) -> None:
    """Render a 'Date Range' aggregation tab for a gamelog page."""
    left, right = st.columns([1, 3])
    with left:
        level = st.selectbox(
            "Level", list(level_map.keys()), index=0, key=f"{key_prefix}_level"
        )
        season = st.multiselect(
            "Season", season_options(full_df),
            default=(
                [season_options(full_df)[1]]
                if len(season_options(full_df)) > 1 else ["All"]
            ),
            key=f"{key_prefix}_season",
        )
        game_type_group = st.selectbox(
            "Game Type", game_type_group_options(full_df),
            index=0, key=f"{key_prefix}_gtg",
        )
        base = full_df[full_df["level_id"].isin(level_map[level])]
        base = filter_by_values(base, "season", season)
        base = filter_by_game_type_group(base, game_type_group)

        team = st.selectbox(
            "Team",
            _clean_team_options(base, team_col, include_all=(entity == "player")),
            index=0, key=f"{key_prefix}_team",
        )
        base = filter_by_team_token(base, team_col, team)

        player_choice = None
        if entity == "player":
            player_opts, player_name_map = player_id_options(base, id_col, name_col)
            player_choice = st.selectbox(
                "Player", player_opts,
                index=0,
                format_func=lambda v: (
                    "All" if v == "All"
                    else f"{player_name_map.get(v, 'Unknown')} ({int(v)})"
                ),
                key=f"{key_prefix}_player",
            )
            base = filter_by_values(base, id_col, player_choice)

        if pitch_tag_filter and "pitch_tag" in base.columns:
            tags = ["All"] + sorted(base["pitch_tag"].dropna().unique().tolist())
            tag_choice = st.multiselect(
                "Pitch Type", tags, default=["All"], key=f"{key_prefix}_tag"
            )
            base = filter_by_values(base, "pitch_tag", tag_choice)

        _dates = pd.to_datetime(base["game_date"], errors="coerce").dropna()
        if _dates.empty:
            st.info("No games available for this filter.")
            with right:
                st.info("Adjust filters to select a date range.")
            return
        min_d, max_d = _dates.min().date(), _dates.max().date()
        start_date = st.date_input(
            "Start date", value=min_d, min_value=min_d, max_value=max_d,
            key=f"{key_prefix}_start",
        )
        end_date = st.date_input(
            "End date", value=max_d, min_value=min_d, max_value=max_d,
            key=f"{key_prefix}_end",
        )
        mode = st.radio(
            "Values", ["Raw counts", "Rates"], horizontal=True,
            key=f"{key_prefix}_mode",
        )

    with right:
        if start_date > end_date:
            st.warning("Start date is after end date.")
            return
        _gd = pd.to_datetime(base["game_date"], errors="coerce")
        in_range = base[
            (_gd.dt.date >= start_date) & (_gd.dt.date <= end_date)
        ].copy()
        if in_range.empty:
            st.info("No games in the selected date range.")
            return

        span_label = f"TOTAL {start_date.strftime('%m/%d')}–{end_date.strftime('%m/%d')}"
        label_col = "Date"
        table = build_span_table(in_range, spec, mode, label_col, span_label)
        if table.empty:
            st.info("Nothing to aggregate.")
            return
        table = table.rename(columns=rename_map)
        if mode == "Rates":
            st.caption(
                "Aggregate row reflects rates computed from summed counts over the span. "
                "Only rates derivable from stored gamelog counts are shown."
            )
        render_table(table, stats_df=pd.DataFrame())
        download_button(table, f"{key_prefix}_span", f"{key_prefix}_span_dl")

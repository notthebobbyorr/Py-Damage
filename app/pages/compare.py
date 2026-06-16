from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.config import RADAR_TEMPLATES
from app.datasets import (
    damage_df,
    hitters_reg_df,
    pitcher_df,
    pitchers_reg_df,
)

_MAX_SELECTIONS = 4
_TRACE_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]


def _dataset_and_ids(
    player_type: str, values: str
) -> tuple[pd.DataFrame, str, str]:
    """Return (df, id_col, name_col)."""
    if player_type == "Hitter":
        if values == "Auto-Regressed":
            return hitters_reg_df, "batter_mlbid", "hitter_name"
        return damage_df, "batter_mlbid", "hitter_name"
    if values == "Auto-Regressed":
        return pitchers_reg_df, "pitcher_mlbid", "name"
    return pitcher_df, "pitcher_mlbid", "name"


def _template_for(
    player_type: str, template_name: str, values: str
) -> list[tuple[str, str, bool]]:
    """Return the spoke list for the chosen template + values mode."""
    mode_key = "ar" if values == "Auto-Regressed" else "primary"
    return RADAR_TEMPLATES[player_type][template_name][mode_key]


def _filter_to_regular_season(df: pd.DataFrame) -> pd.DataFrame:
    if "game_type_group" in df.columns:
        return df[df["game_type_group"] == "Regular Season"]
    return df


def _filter_to_mlb(df: pd.DataFrame) -> pd.DataFrame:
    if "level_id" in df.columns:
        mask = pd.to_numeric(df["level_id"], errors="coerce") == 1
        return df[mask]
    return df


@st.cache_resource
def _percentile_basis(
    player_type: str, values: str
) -> dict[int, pd.DataFrame]:
    """Per-season MLB Regular-Season pool, indexed by season. Caches the full
    dataset slice so any template's spoke columns can be looked up against it."""
    df, _id_col, _name_col = _dataset_and_ids(player_type, values)
    if df.empty:
        return {}
    base = _filter_to_mlb(_filter_to_regular_season(df))
    if base.empty:
        return {}
    out: dict[int, pd.DataFrame] = {}
    for season, grp in base.groupby("season", observed=True):
        try:
            season_key = int(season)
        except (TypeError, ValueError):
            continue
        out[season_key] = grp.reset_index(drop=True)
    return out


def _percentile_for_row(
    row: pd.Series, pool: pd.DataFrame, template: list, reverse: bool
) -> list[tuple[str, float | None]]:
    """Return [(display_label, percentile_0_to_100 or None)] in template order."""
    results: list[tuple[str, float | None]] = []
    for col, label, higher_is_worse in template:
        if col not in pool.columns or col not in row.index:
            results.append((label, None))
            continue
        val = row[col]
        if pd.isna(val):
            results.append((label, None))
            continue
        series = pd.to_numeric(pool[col], errors="coerce").dropna()
        if series.empty:
            results.append((label, None))
            continue
        pct = float((series <= float(val)).mean() * 100.0)
        if higher_is_worse:
            pct = 100.0 - pct
        results.append((label, pct))
    return results


def _player_options(df: pd.DataFrame, id_col: str, name_col: str) -> tuple[list, dict]:
    if df.empty or id_col not in df.columns or name_col not in df.columns:
        return [], {}
    pool = _filter_to_mlb(_filter_to_regular_season(df))
    if pool.empty:
        pool = df
    pairs = (
        pool[[id_col, name_col]]
        .dropna()
        .drop_duplicates(subset=[id_col])
    )
    pairs[id_col] = pd.to_numeric(pairs[id_col], errors="coerce")
    pairs = pairs.dropna(subset=[id_col])
    pairs[id_col] = pairs[id_col].astype(int)
    pairs = pairs.sort_values(name_col)
    ids = pairs[id_col].tolist()
    name_map = dict(zip(pairs[id_col].tolist(), pairs[name_col].astype(str).tolist()))
    return ids, name_map


def _seasons_for_player(df: pd.DataFrame, id_col: str, player_id) -> list[int]:
    if df.empty or "season" not in df.columns:
        return []
    pool = _filter_to_mlb(_filter_to_regular_season(df))
    rows = pool[pool[id_col] == player_id]
    seasons = pd.to_numeric(rows["season"], errors="coerce").dropna().astype(int)
    return sorted(seasons.unique().tolist(), reverse=True)


def _row_for(
    df: pd.DataFrame, id_col: str, player_id, season: int
) -> pd.Series | None:
    pool = _filter_to_mlb(_filter_to_regular_season(df))
    season_match = pd.to_numeric(pool["season"], errors="coerce") == int(season)
    rows = pool[(pool[id_col] == player_id) & season_match]
    if rows.empty:
        return None
    return rows.iloc[0]


def radar_compare():
    """Compare - Radar Plot Comparisons page"""
    st.title("Radar Plot Comparisons")
    st.caption(
        "Compare up to 4 player-seasons across template metric sets. "
        "Each spoke is a league-wide percentile (MLB, Regular Season). "
        "Spokes for higher-is-worse metrics are inverted so a larger area is always better."
    )

    top1, top2, top3 = st.columns([1, 1, 2])
    with top1:
        player_type = st.radio(
            "Player type",
            list(RADAR_TEMPLATES.keys()),
            horizontal=True,
            key="radar_player_type",
        )
    with top2:
        values_mode = st.radio(
            "Values",
            ["Raw", "Auto-Regressed"],
            horizontal=True,
            key="radar_values_mode",
        )
    with top3:
        template_options = list(RADAR_TEMPLATES[player_type].keys())
        template_name = st.selectbox(
            "Template",
            options=template_options,
            index=0,
            key=f"radar_template_{player_type}",
        )

    df, id_col, name_col = _dataset_and_ids(player_type, values_mode)
    if df.empty:
        st.info(f"No {player_type} data loaded.")
        return
    template = _template_for(player_type, template_name, values_mode)
    st.caption(f"Template: **{template_name}** ({len(template)} metrics)")

    ids, name_map = _player_options(df, id_col, name_col)
    if not ids:
        st.info("No eligible MLB players found.")
        return

    pool_by_season = _percentile_basis(player_type, values_mode)

    selections: list[tuple[int, int]] = []
    st.markdown("**Select up to 4 player-seasons**")
    for i in range(_MAX_SELECTIONS):
        row_col1, row_col2 = st.columns([2, 1])
        with row_col1:
            options = ["(none)"] + ids
            choice = st.selectbox(
                f"Player {i + 1}",
                options=options,
                index=0,
                format_func=lambda v: (
                    "(none)"
                    if v == "(none)"
                    else f"{name_map.get(int(v), 'Unknown')} ({int(v)})"
                ),
                key=f"radar_player_{i}",
            )
        with row_col2:
            if choice == "(none)":
                st.selectbox(
                    f"Season {i + 1}", options=["—"], index=0, key=f"radar_season_{i}",
                    disabled=True,
                )
                continue
            seasons = _seasons_for_player(df, id_col, int(choice))
            if not seasons:
                st.selectbox(
                    f"Season {i + 1}", options=["(none)"], index=0,
                    key=f"radar_season_{i}",
                )
                continue
            season_choice = st.selectbox(
                f"Season {i + 1}",
                options=seasons,
                index=0,
                key=f"radar_season_{i}",
            )
            selections.append((int(choice), int(season_choice)))

    if not selections:
        st.info("Pick at least one player-season above to render the radar.")
        return

    fig = go.Figure()
    spoke_labels = [label for _, label, _ in template]
    rendered_any = False
    for idx, (pid, season) in enumerate(selections):
        row = _row_for(df, id_col, pid, season)
        if row is None:
            continue
        pool = pool_by_season.get(season)
        if pool is None or pool.empty:
            continue
        pct_pairs = _percentile_for_row(row, pool, template, reverse=False)
        spoke_values = [v if v is not None else 0.0 for _, v in pct_pairs]
        # Close the polygon by repeating the first point
        radial = spoke_values + [spoke_values[0]]
        theta = spoke_labels + [spoke_labels[0]]
        trace_name = f"{name_map.get(pid, 'Unknown')} ({season})"
        color = _TRACE_COLORS[idx % len(_TRACE_COLORS)]
        fig.add_trace(
            go.Scatterpolar(
                r=radial,
                theta=theta,
                fill="toself",
                name=trace_name,
                line=dict(color=color, width=2),
                opacity=0.45,
            )
        )
        rendered_any = True

    if not rendered_any:
        st.info("Selected players/seasons had no data in the percentile pool.")
        return

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], tickfont=dict(size=10)),
            angularaxis=dict(tickfont=dict(size=11)),
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=30, b=60),
        height=620,
    )
    st.plotly_chart(fig, width="stretch", key="radar_compare_chart")

    st.markdown("---")
    st.markdown("**Underlying percentile values**")
    table_rows = []
    for pid, season in selections:
        row = _row_for(df, id_col, pid, season)
        if row is None:
            continue
        pool = pool_by_season.get(season)
        if pool is None:
            continue
        pcts = _percentile_for_row(row, pool, template, reverse=False)
        rec = {"Player": f"{name_map.get(pid, 'Unknown')} ({season})"}
        for label, val in pcts:
            rec[label] = round(val, 0) if val is not None else None
        table_rows.append(rec)
    if table_rows:
        pct_df = pd.DataFrame(table_rows).set_index("Player")
        st.dataframe(pct_df, width="stretch")

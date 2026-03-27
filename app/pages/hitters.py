from __future__ import annotations

import numpy as np
import streamlit as st

from app.config import (
    ABS_GRADIENT_COLS_PITCHERS,
    GAME_TYPE_GROUP_NOTE,
    HIGHER_IS_WORSE_COLS,
    HITTER_COMPS_BASE_FEATURE_COLS,
    HITTER_COMPS_EXTRA_FEATURE_COLS,
    LEVEL_LABELS,
    POSITION_FILTER_LABELS,
)
from app.datasets import (
    damage_df,
    hitter_mlb_eq_coeffs,
    hitter_mlb_eq_metrics,
    hitter_pct,
    hitter_splits_df,
    hitters_mlb_eq_df,
    hitters_reg_df,
)
from app.filters import (
    download_button,
    filter_by_game_type_group,
    filter_by_positions,
    filter_by_team_token,
    filter_by_values,
    game_type_group_options,
    numeric_filter,
    player_id_options,
    position_options,
    season_options,
    team_options,
)
from app.utils import _hitter_display_map, _similarity_choice_labels
from app.viz import render_table


def hitter_individual_stats():
    """Hitters - Individual Stats page"""
    st.title("Individual Hitter Stats")

    if damage_df.empty:
        st.info("Missing hitter damage data file.")
    else:
        left, right = st.columns([1, 3])
        with left:
            level = st.selectbox(
                "Select Level",
                ["All", "MLB", "Triple-A", "Low-A", "Low Minors"],
                index=1,
                key="hitter_stats_level",
            )
            season = st.multiselect(
                "Select Season",
                season_options(damage_df),
                default=(
                    [season_options(damage_df)[1]]
                    if len(season_options(damage_df)) > 1
                    else ["All"]
                ),
                key="hitter_stats_season",
            )
            game_type_group = st.selectbox(
                "Game Type",
                game_type_group_options(damage_df),
                index=0,
                key="hitter_stats_game_type_group",
            )
            min_value = st.number_input(
                "Minimum Value",
                min_value=0,
                max_value=500,
                value=100,
                step=1,
                key="hitter_stats_min_value",
            )
            value_type = st.selectbox(
                "Filter By", ["PA", "BBE"], index=1, key="hitter_stats_value_type"
            )
            team = st.selectbox(
                "Select Team",
                team_options(damage_df, "hitting_code"),
                index=0,
                key="hitter_stats_team",
            )
            position = st.multiselect(
                "Select Position",
                position_options(damage_df),
                default=["All"],
                key="hitter_stats_position",
                format_func=lambda v: (
                    "All" if v == "All" else POSITION_FILTER_LABELS.get(v, v)
                ),
            )
            player_options_list, player_name_map = player_id_options(
                damage_df, "batter_mlbid", "hitter_name"
            )
            player = st.multiselect(
                "Select Player",
                player_options_list,
                default=["All"],
                format_func=lambda v: (
                    "All"
                    if v == "All"
                    else f"{player_name_map.get(v, 'Unknown')} ({int(v)})"
                ),
                key="hitter_stats_player",
            )
        with right:
            level_map = {
                "All": [1, 11, 14, 16],
                "MLB": [1],
                "Triple-A": [11],
                "Low-A": [14],
                "Low Minors": [16],
            }
            base_stats = damage_df.copy()
            base_stats = base_stats.assign(
                __season=base_stats["season"], __level=base_stats["level_id"]
            )
            if game_type_group != "Regular Season":
                st.info(GAME_TYPE_GROUP_NOTE.format(game_type_group))
            df = damage_df.copy()
            df = df[df["level_id"].isin(level_map[level])]
            df = filter_by_values(df, "season", season)
            df = filter_by_game_type_group(df, game_type_group)
            df = filter_by_team_token(df, "hitting_code", team)
            df = filter_by_positions(df, position)
            df = filter_by_values(df, "batter_mlbid", player)
            df = df.assign(__season=df["season"], __level=df["level_id"])

            if value_type == "PA":
                df = numeric_filter(df, "PA", min_value)
            else:
                df = numeric_filter(df, "bbe", min_value)

            columns = [
                "hitter_name", "batter_mlbid", "hitting_code", "season", "PA", "bbe", "HR",
                "damage_rate", "EV90th", "max_EV", "pull_FB_pct", "LA_gte_20",
                "LA_lte_0", "SEAGER", "selection_skill", "hittable_pitches_taken",
                "chase", "z_con", "secondary_whiff_pct", "whiffs_vs_95",
                "contact_vs_avg",
                "Swing_pct", "p_Swing_with_loc_pct",
                "__season", "__level",
            ]
            df = df[[col for col in columns if col in df.columns]].copy()
            rename_map = {
                "hitter_name": "Name", "batter_mlbid": "Player ID",
                "hitting_code": "Team", "season": "Season", "bbe": "BBE", "HR": "HR",
                "damage_rate": "Damage/BBE (%)", "EV90th": "90th Pctile EV",
                "max_EV": "Max EV", "pull_FB_pct": "Pulled FB (%)",
                "LA_gte_20": "LA>=20%", "LA_lte_0": "LA<=0%",
                "selection_skill": "Selectivity (%)",
                "hittable_pitches_taken": "Hittable Pitch Take (%)",
                "chase": "Chase (%)", "z_con": "Z-Contact (%)",
                "secondary_whiff_pct": "Whiff vs. Secondaries (%)",
                "whiffs_vs_95": "Whiff vs. 95+ (%)",
                "contact_vs_avg": "Contact Over Expected (%)",
                "Swing_pct": "Swing (%)",
                "p_Swing_with_loc_pct": "pSwing (%)",
            }
            df = df.rename(columns=rename_map)
            df = df.sort_values(by="Damage/BBE (%)", ascending=False)
            stats_df = base_stats[
                [col for col in columns if col in base_stats.columns]
            ].rename(columns=rename_map)
            render_table(
                df,
                reverse_cols=HIGHER_IS_WORSE_COLS | {"Chase (%)", "LA<=0%"},
                group_cols=["__season", "__level"],
                stats_df=stats_df,
                include_team_label=False,
            )
            download_button(df, "hitters", "hitters_download")


def hitter_percentiles():
    """Hitters - Percentiles page"""
    st.title("Hitter Percentiles")

    if hitter_pct.empty:
        st.info("Missing hitter_pctiles.csv")
    else:
        left, right = st.columns([1, 3])
        with left:
            level = st.selectbox(
                "Select Level",
                ["All", "MLB", "Triple-A", "Low-A", "Low Minors"],
                index=1,
                key="hitter_pct_level",
            )
            season = st.multiselect(
                "Select Season",
                season_options(hitter_pct),
                default=(
                    [season_options(hitter_pct)[1]]
                    if len(season_options(hitter_pct)) > 1
                    else ["All"]
                ),
                key="hitter_pct_season",
            )
            game_type_group = st.selectbox(
                "Game Type",
                game_type_group_options(hitter_pct),
                index=0,
                key="hitter_pct_game_type_group",
            )
            min_value = st.number_input(
                "Minimum Value",
                min_value=0,
                max_value=500,
                value=100,
                step=1,
                key="hitter_pct_min_value",
            )
            value_type = st.selectbox(
                "Filter By", ["PA", "BBE"], index=1, key="hitter_pct_value_type"
            )
            team = st.selectbox(
                "Select Team",
                team_options(hitter_pct, "hitting_code"),
                index=0,
                key="hitter_pct_team",
            )
            position = st.multiselect(
                "Select Position",
                position_options(hitter_pct),
                default=["All"],
                key="hitter_pct_position",
                format_func=lambda v: (
                    "All" if v == "All" else POSITION_FILTER_LABELS.get(v, v)
                ),
            )
            player_options_list, player_name_map = player_id_options(
                hitter_pct, "batter_mlbid", "hitter_name"
            )
            player = st.multiselect(
                "Select Player",
                player_options_list,
                default=["All"],
                format_func=lambda v: (
                    "All"
                    if v == "All"
                    else f"{player_name_map.get(v, 'Unknown')} ({int(v)})"
                ),
                key="hitter_pct_player",
            )
        with right:
            if game_type_group != "Regular Season":
                st.info(GAME_TYPE_GROUP_NOTE.format(game_type_group))
            level_map = {
                "All": [1, 11, 14, 16],
                "MLB": [1],
                "Triple-A": [11],
                "Low-A": [14],
                "Low Minors": [16],
            }
            df = hitter_pct.copy()
            df = df[df["level_id"].isin(level_map[level])]
            df = filter_by_values(df, "season", season)
            df = filter_by_game_type_group(df, game_type_group)
            df = filter_by_team_token(df, "hitting_code", team)
            df = filter_by_positions(df, position)
            df = filter_by_values(df, "batter_mlbid", player)

            if value_type == "PA":
                df = numeric_filter(df, "PA", min_value)
            else:
                df = numeric_filter(df, "bbe", min_value)

            columns = [
                "hitter_name", "batter_mlbid", "hitting_code", "season", "HR",
                "SEAGER_pctile", "selection_skill_pctile", "hittable_pitches_taken_pctile",
                "damage_rate_pctile", "EV90th_pctile", "max_EV_pctile",
                "pull_FB_pct_pctile", "chase_pctile", "z_con_pctile",
                "secondary_whiff_pct_pctile", "whiffs_vs_95_pctile",
                "contact_vs_avg_pctile", "__season", "__level",
            ]
            df = df.assign(__season=df["season"], __level=df["level_id"])
            df = df[[col for col in columns if col in df.columns]].copy()
            rename_map = {
                "hitter_name": "Name", "batter_mlbid": "Player ID",
                "hitting_code": "Team", "season": "Season", "HR": "HR",
                "SEAGER_pctile": "SEAGER", "selection_skill_pctile": "Selection Skill",
                "hittable_pitches_taken_pctile": "Hittable Pitch Take",
                "damage_rate_pctile": "Damage Rate", "EV90th_pctile": "90th Pctile EV",
                "max_EV_pctile": "Max EV", "pull_FB_pct_pctile": "Pulled FB",
                "chase_pctile": "Chase", "z_con_pctile": "Z-Contact",
                "secondary_whiff_pct_pctile": "Whiff vs Secondaries",
                "whiffs_vs_95_pctile": "Whiff vs 95+",
                "contact_vs_avg_pctile": "Contact Over Expected",
            }
            df = df.rename(columns=rename_map)
            df = df.sort_values(by="Damage Rate", ascending=False)
            render_table(
                df,
                reverse_cols={
                    "Hittable Pitch Take",
                    "Chase",
                    "Whiff vs Secondaries",
                    "Whiff vs 95+",
                },
            )
            download_button(df, "hitter_percentiles", "hitter_pct_download")


def hitter_comps():
    """Hitters - Comparisons page"""
    st.title("Hitter Comparisons (Auto-Regressed)")

    if hitters_reg_df.empty:
        st.info("Missing hitters_regressed.csv")
        return

    use_mlb_eq = st.toggle(
        "Use MLB-equivalent translated stats",
        value=False,
        key="hitter_comps_use_mlb_eq",
        help=(
            "Use intra+inter-season level translations to compare non-MLB seasons "
            "against MLB seasons in MLB-equivalent space."
        ),
    )
    comp_df = hitters_reg_df.copy()
    if use_mlb_eq:
        if hitters_mlb_eq_df.empty:
            st.info(
                "MLB-equivalent translation table is unavailable; using raw AR stats."
            )
            use_mlb_eq = False
        else:
            comp_df = hitters_mlb_eq_df.copy()

    if use_mlb_eq:
        player_pool = comp_df[(comp_df["PA"] >= 20)].copy()
        eligible_all = comp_df[
            (comp_df["level_id"] == 1) & (comp_df["PA"] >= 200)
        ].copy()
        if player_pool.empty:
            st.info("No eligible hitter seasons (min 20 PA).")
            return
        target_levels = sorted(player_pool["level_id"].dropna().unique().tolist())
        if not target_levels:
            st.info("No target levels available.")
            return
        target_level = st.selectbox(
            "Target Level",
            target_levels,
            index=0,
            key="hitter_comps_target_level",
            format_func=lambda v: LEVEL_LABELS.get(int(v), str(int(v))),
        )
        player_pool = player_pool[player_pool["level_id"] == target_level]
    else:
        player_pool = comp_df[(comp_df["level_id"] == 1) & (comp_df["PA"] >= 20)].copy()
        eligible_all = comp_df[
            (comp_df["level_id"] == 1) & (comp_df["PA"] >= 200)
        ].copy()

    position = st.multiselect(
        "Select Position",
        position_options(player_pool),
        default=["All"],
        key="hitter_comps_position",
        format_func=lambda v: (
            "All" if v == "All" else POSITION_FILTER_LABELS.get(v, v)
        ),
    )
    player_pool = filter_by_positions(player_pool, position)
    eligible_all = filter_by_positions(eligible_all, position)

    if player_pool.empty:
        st.info("No eligible MLB hitter seasons (min 20 PA).")
        return
    if eligible_all.empty:
        st.info("No eligible MLB comparison seasons (min 200 PA).")
        return

    seasons = season_options(player_pool, "season")[1:]
    if not seasons:
        st.info("No seasons available for this view.")
        return
    season_choice = st.selectbox("Season", seasons, index=0, key="hitter_comps_season")
    season_df = player_pool[player_pool["season"] == season_choice]
    if season_df.empty:
        st.info("No player rows for this season selection.")
        return

    player_options_list, player_name_map = player_id_options(
        season_df, "batter_mlbid", "hitter_name"
    )
    player_values = [opt for opt in player_options_list if opt != "All"]
    if not player_values:
        st.info("No players available for this filter.")
        return
    player_choice = st.selectbox(
        "Player",
        player_values,
        index=0,
        format_func=lambda v: f"{player_name_map.get(v, 'Unknown')} ({int(v)})",
        key="hitter_comps_player",
    )
    player_df = season_df[season_df["batter_mlbid"] == player_choice]
    teams = team_options(player_df, "hitting_code")
    team_choice = st.selectbox("Team", teams, index=0, key="hitter_comps_team")
    if team_choice and team_choice != "All":
        filtered_player_df = filter_by_team_token(
            player_df, "hitting_code", team_choice
        )
        if filtered_player_df.empty:
            st.warning(
                "No rows for that team at this level/season; using all team rows for selection."
            )
        else:
            player_df = filtered_player_df

    metric_suffix = "_mlb_eq" if use_mlb_eq else ""
    default_feature_cols = [
        f"{col}{metric_suffix}" for col in HITTER_COMPS_BASE_FEATURE_COLS
    ]
    allowed_cols = list(
        dict.fromkeys(
            default_feature_cols
            + [f"{col}{metric_suffix}" for col in HITTER_COMPS_EXTRA_FEATURE_COLS]
        )
    )

    display_map = _hitter_display_map(include_mlb_eq=False)
    if use_mlb_eq:
        base_display_map = _hitter_display_map(include_mlb_eq=False)
        for col, label in base_display_map.items():
            if col.endswith("_reg"):
                display_map[f"{col}_mlb_eq"] = label
    exclude_cols = {
        "batter_mlbid", "pitcher_mlbid", "level_id", "game_pk", "PA", "IP", "TBF",
        "GS", "pitches", "pitches_n", "pitches_num", "pitches_den", "bbe", "season",
        "lg_contact_baseline",
    }
    numeric_cols, similarity_labels = _similarity_choice_labels(
        eligible_all, display_map, exclude_cols
    )
    numeric_cols = [col for col in numeric_cols if col in allowed_cols]
    default_feature_cols = [col for col in default_feature_cols if col in numeric_cols]

    similarity_key = (
        "hitter_comps_similarity_cols_mlb_eq"
        if use_mlb_eq
        else "hitter_comps_similarity_cols_raw"
    )
    feature_cols = st.multiselect(
        "Similarity Score Columns",
        options=numeric_cols,
        default=default_feature_cols,
        key=similarity_key,
        format_func=lambda col: similarity_labels.get(col, col),
    )
    feature_cols = [col for col in feature_cols if col in numeric_cols]
    feature_cols = list(dict.fromkeys(feature_cols))
    if not feature_cols:
        st.info("Select at least one column to compute similarity scores.")
        return
    if player_df.empty:
        st.info("No season row found for that selection.")
        return

    eligible_comp = eligible_all.copy()
    eligible_comp = eligible_comp[~(eligible_comp["batter_mlbid"] == player_choice)]
    eligible_comp = eligible_comp[eligible_comp[feature_cols].notna().any(axis=1)]
    if eligible_comp.empty:
        st.info("No comparable MLB rows found after filters.")
        return

    stats = eligible_comp[feature_cols].copy()
    means = stats.mean().fillna(0.0)
    stats = stats.fillna(means)
    stds = stats.std(ddof=0).replace(0, np.nan)
    zscores = ((stats - means) / stds).fillna(0)
    target_stats = player_df[feature_cols].copy().fillna(means)
    target_vec = ((target_stats - means) / stds).fillna(0).iloc[0].to_numpy()
    distances = np.linalg.norm(zscores.to_numpy() - target_vec, axis=1)
    max_dist = distances.max() if len(distances) else 0.0
    if max_dist == 0:
        similarity = np.full_like(distances, 100.0, dtype=float)
    else:
        similarity = 100 * (1 - (distances / max_dist))

    eligible_comp = eligible_comp.copy()
    eligible_comp["similarity_score"] = similarity.round(0)
    eligible_comp = eligible_comp.sort_values("similarity_score", ascending=False)
    eligible_comp = eligible_comp.assign(
        __season=eligible_comp["season"], __level=eligible_comp["level_id"]
    )

    base_rename = {
        "hitter_name": "Name", "hitting_code": "Team", "season": "Season",
        "bbe": "BBE", "similarity_score": "Similarity (0-100)",
    }
    display_cols = [
        "hitter_name", "hitting_code", "season", "PA", "bbe", "similarity_score",
        *feature_cols, "__season", "__level",
    ]
    df = eligible_comp[
        [col for col in display_cols if col in eligible_comp.columns]
    ].copy()
    df = df.rename(columns={**base_rename, **similarity_labels})
    df = df.loc[:, ~df.columns.duplicated()]

    stats_df = eligible_all.copy()
    stats_df = stats_df.assign(
        __season=stats_df["season"], __level=stats_df["level_id"]
    )
    stats_columns = [
        "hitter_name", "hitting_code", "season", "PA", "bbe",
        *list(dict.fromkeys(default_feature_cols + feature_cols)),
        "__season", "__level",
    ]
    stats_df = stats_df[
        [col for col in stats_columns if col in stats_df.columns]
    ].rename(columns={**base_rename, **similarity_labels})
    stats_df = stats_df.loc[:, ~stats_df.columns.duplicated()]

    target_cols = [
        "hitter_name", "hitting_code", "season", "PA", "bbe",
        *list(dict.fromkeys(default_feature_cols + feature_cols)),
        "__season", "__level",
    ]
    target_df = player_df.assign(
        __season=player_df["season"], __level=player_df["level_id"]
    )
    target_df = target_df[
        [col for col in target_cols if col in target_df.columns]
    ].copy()
    if use_mlb_eq and "__level" in target_df.columns:
        target_df["__level"] = 1
    target_df = target_df.rename(columns={**base_rename, **similarity_labels})
    target_df = target_df.loc[:, ~target_df.columns.duplicated()]

    reverse_hitters = HIGHER_IS_WORSE_COLS | {
        "LA<=0%", "Chase (%)", "Swing Length", "Attack Angle", "VBA",
    }

    if use_mlb_eq:
        level_label = LEVEL_LABELS.get(int(target_level), str(int(target_level)))
        player_name = str(
            player_df["hitter_name"].iloc[0]
            if "hitter_name" in player_df.columns and not player_df.empty
            else player_name_map.get(player_choice, "Player")
        )
        st.caption(
            f"{player_name}'s MLB-equivalent statistics derived from their "
            f"{level_label} statistics:"
        )
    else:
        st.caption("Selected season")
    render_table(
        target_df,
        reverse_cols=reverse_hitters,
        group_cols=["__season", "__level"],
        stats_df=stats_df,
        show_controls=False,
        hide_cols={"Team"},
    )
    if use_mlb_eq:
        st.caption(
            "Most similar MLB seasons by translated MLB-equivalent stats (PA >= 200)"
        )
    else:
        st.caption("Most similar MLB seasons (PA >= 200)")
    render_table(
        df,
        reverse_cols=reverse_hitters,
        group_cols=["__season", "__level"],
        stats_df=stats_df,
    )


def hitter_mlb_equivalencies():
    """Hitters - MLB equivalency translations (intra+inter-season chained)."""
    st.title("Hitter MLB Equivalencies")

    if hitters_mlb_eq_df.empty:
        st.info("MLB-equivalent table is unavailable.")
        return

    st.caption(
        "Intra+inter-season chained translations in season-adjusted z-score space "
        "(16->14->11->1)."
    )
    st.caption(
        "Final translated values also apply directional minimum-shift calibration "
        "for key hitter metrics."
    )
    st.caption(
        "Regression fits are trained on both same-season and season n->n+1 "
        "level transitions."
    )

    view = hitters_mlb_eq_df.copy()
    if "PA" in view.columns:
        min_pa = st.number_input(
            "Minimum PA",
            min_value=0,
            max_value=1000,
            value=20,
            step=5,
            key="hitter_mlb_eq_min_pa",
        )
        view = view[view["PA"] >= min_pa]

    season_vals = season_options(view)
    season = st.selectbox(
        "Season",
        season_vals,
        index=(1 if len(season_vals) > 1 else 0),
        key="hitter_mlb_eq_season",
    )
    view = filter_by_values(view, "season", season)

    level_options = ["All", "MLB", "Triple-A", "Low-A", "Low Minors"]
    level_choice = st.selectbox(
        "Level", level_options, index=0, key="hitter_mlb_eq_level",
    )
    level_map = {
        "All": [1, 11, 14, 16], "MLB": [1], "Triple-A": [11],
        "Low-A": [14], "Low Minors": [16],
    }
    view = view[view["level_id"].isin(level_map[level_choice])]

    team = st.selectbox(
        "Team", team_options(view, "hitting_code"), index=0, key="hitter_mlb_eq_team",
    )
    view = filter_by_team_token(view, "hitting_code", team)
    position = st.multiselect(
        "Position", position_options(view), default=["All"],
        key="hitter_mlb_eq_position",
        format_func=lambda v: (
            "All" if v == "All" else POSITION_FILTER_LABELS.get(v, v)
        ),
    )
    view = filter_by_positions(view, position)
    if view.empty:
        st.info("No rows after filtering.")
        return

    metric_base_cols = [
        col
        for col in HITTER_COMPS_BASE_FEATURE_COLS + HITTER_COMPS_EXTRA_FEATURE_COLS
        if col in view.columns and f"{col}_mlb_eq" in view.columns
    ]
    default_metrics = [
        col for col in HITTER_COMPS_BASE_FEATURE_COLS if col in metric_base_cols
    ]
    selected_metrics = st.multiselect(
        "Metric Columns", options=metric_base_cols, default=default_metrics,
        key="hitter_mlb_eq_metrics",
        format_func=lambda col: _hitter_display_map().get(col, col),
    )
    if not selected_metrics:
        st.info("Select at least one metric column.")
        return

    table_df = view.copy()
    for col in selected_metrics:
        eq_col = f"{col}_mlb_eq"
        delta_col = f"{col}_mlb_delta"
        table_df[delta_col] = table_df[eq_col] - table_df[col]
    table_df = table_df.assign(
        Level=table_df["level_id"].map(lambda v: LEVEL_LABELS.get(int(v), str(int(v)))),
        __season=table_df["season"],
        __level=table_df["level_id"],
    )

    metric_cols: list[str] = []
    for col in selected_metrics:
        metric_cols.extend([col, f"{col}_mlb_eq", f"{col}_mlb_delta"])

    show_cols = [
        "hitter_name", "batter_mlbid", "hitting_code", "season", "Level", "PA", "bbe",
        *metric_cols, "__season", "__level",
    ]
    show_cols = [col for col in show_cols if col in table_df.columns]
    table_df = table_df[show_cols].copy()

    rename_map = {
        "hitter_name": "Name", "batter_mlbid": "Player ID",
        "hitting_code": "Team", "season": "Season", "bbe": "BBE",
    }
    display_map = _hitter_display_map(include_mlb_eq=True)
    for col in selected_metrics:
        rename_map[col] = display_map.get(col, col)
        rename_map[f"{col}_mlb_eq"] = display_map.get(f"{col}_mlb_eq", f"{col} MLB Eq")
        rename_map[f"{col}_mlb_delta"] = (
            f"{display_map.get(f'{col}_mlb_eq', f'{col} MLB Eq')} Delta"
        )
    table_df = table_df.rename(columns=rename_map)

    reverse_cols = HIGHER_IS_WORSE_COLS | {"LA<=0%", "Chase (%)"}
    reverse_cols = reverse_cols | {f"{name} MLB Eq" for name in reverse_cols}
    render_table(
        table_df,
        reverse_cols=reverse_cols,
        group_cols=["__season", "__level"],
        stats_df=table_df,
    )
    download_button(table_df, "hitter_mlb_equivalencies", "hitter_mlb_eq_download")

    if not hitter_mlb_eq_coeffs.empty:
        with st.expander("Translation coefficients", expanded=False):
            coeff_df = hitter_mlb_eq_coeffs.copy()
            coeff_df = coeff_df.assign(
                from_level=coeff_df["src_level"].map(
                    lambda v: LEVEL_LABELS.get(int(v), str(int(v)))
                ),
                to_level=coeff_df["dst_level"].map(
                    lambda v: LEVEL_LABELS.get(int(v), str(int(v)))
                ),
            )
            coeff_df["metric"] = coeff_df["metric"].map(
                lambda c: _hitter_display_map().get(c, c)
            )
            coeff_df = coeff_df.rename(
                columns={
                    "metric": "Metric", "from_level": "From", "to_level": "To",
                    "a": "Intercept (a)", "b": "Rate (b)", "n": "Sample",
                    "fit_type": "Type", "min_src_pa": "Min PA (From)",
                    "min_dst_pa": "Min PA (To)",
                }
            )
            coeff_cols = [
                "Metric", "From", "To", "Type", "Intercept (a)", "Rate (b)",
                "Sample", "Min PA (From)", "Min PA (To)",
            ]
            coeff_df = coeff_df[[col for col in coeff_cols if col in coeff_df.columns]]
            render_table(coeff_df, show_controls=False, round_decimals=2)


def hitter_ar():
    """Hitters - Auto Regressed page"""
    st.title("Hitters - Auto Regressed")

    if hitters_reg_df.empty:
        st.info("Missing hitters_regressed.csv or hitter damage data file.")
    else:
        left, right = st.columns([1, 3])
        with left:
            level = st.selectbox(
                "Select Level",
                ["All", "MLB", "Triple-A", "Low-A", "Low Minors"],
                index=1,
                key="hitter_ar_level",
            )
            season = st.multiselect(
                "Select Season",
                season_options(hitters_reg_df),
                default=(
                    [season_options(hitters_reg_df)[1]]
                    if len(season_options(hitters_reg_df)) > 1
                    else ["All"]
                ),
                key="hitter_ar_season",
            )
            game_type_group = st.selectbox(
                "Game Type",
                game_type_group_options(hitters_reg_df),
                index=0,
                key="hitter_ar_game_type_group",
            )
            min_value = st.number_input(
                "Minimum Value",
                min_value=0,
                max_value=500,
                value=100,
                step=1,
                key="hitter_ar_min_value",
            )
            value_type = st.selectbox(
                "Filter By", ["PA", "BBE"], index=1, key="hitter_ar_value_type"
            )
            team = st.selectbox(
                "Select Team",
                team_options(hitters_reg_df, "hitting_code"),
                index=0,
                key="hitter_ar_team",
            )
            position = st.multiselect(
                "Select Position",
                position_options(hitters_reg_df),
                default=["All"],
                key="hitter_ar_position",
                format_func=lambda v: (
                    "All" if v == "All" else POSITION_FILTER_LABELS.get(v, v)
                ),
            )
            player_options_list, player_name_map = player_id_options(
                hitters_reg_df, "batter_mlbid", "hitter_name"
            )
            player = st.multiselect(
                "Select Player",
                player_options_list,
                default=["All"],
                format_func=lambda v: (
                    "All"
                    if v == "All"
                    else f"{player_name_map.get(v, 'Unknown')} ({int(v)})"
                ),
                key="hitter_ar_player",
            )
        with right:
            if game_type_group != "Regular Season":
                st.info(GAME_TYPE_GROUP_NOTE.format(game_type_group))
            level_map = {
                "All": [1, 11, 14, 16],
                "MLB": [1],
                "Triple-A": [11],
                "Low-A": [14],
                "Low Minors": [16],
            }
            base_stats = hitters_reg_df.copy()
            base_stats = base_stats.assign(
                __season=base_stats["season"], __level=base_stats["level_id"]
            )
            df = hitters_reg_df.copy()
            df = df[df["level_id"].isin(level_map[level])]
            df = filter_by_values(df, "season", season)
            df = filter_by_game_type_group(df, game_type_group)
            df = filter_by_team_token(df, "hitting_code", team)
            df = filter_by_positions(df, position)
            df = filter_by_values(df, "batter_mlbid", player)

            if value_type == "PA":
                df = numeric_filter(df, "PA", min_value)
            else:
                df = numeric_filter(df, "bbe", min_value)

            columns = [
                "hitter_name", "batter_mlbid", "hitting_code", "season", "PA", "bbe", "HR",
                "damage_rate_reg", "EV90th_reg", "max_EV_reg", "pull_FB_pct_reg",
                "LA_gte_20_reg", "LA_lte_0_reg", "SEAGER_reg", "selection_skill_reg",
                "hittable_pitches_taken_reg", "chase_reg", "z_con_reg",
                "secondary_whiff_pct_reg", "whiffs_vs_95_reg", "contact_vs_avg_reg",
                "Swing_pct_reg", "p_Swing_with_loc_pct_reg",
                "__season", "__level",
            ]
            df = df.assign(__season=df["season"], __level=df["level_id"])
            df = df[[col for col in columns if col in df.columns]].copy()
            rename_map = {
                "hitter_name": "Name", "batter_mlbid": "Player ID",
                "hitting_code": "Team", "season": "Season", "bbe": "BBE", "HR": "HR",
                "damage_rate_reg": "Damage/BBE (%)", "EV90th_reg": "90th Pctile EV",
                "max_EV_reg": "Max EV", "pull_FB_pct_reg": "Pulled FB (%)",
                "LA_gte_20_reg": "LA>=20%", "LA_lte_0_reg": "LA<=0%",
                "SEAGER_reg": "SEAGER", "selection_skill_reg": "Selectivity (%)",
                "hittable_pitches_taken_reg": "Hittable Pitch Take (%)",
                "chase_reg": "Chase (%)", "z_con_reg": "Z-Contact (%)",
                "secondary_whiff_pct_reg": "Whiff vs. Secondaries (%)",
                "whiffs_vs_95_reg": "Whiff vs. 95+ (%)",
                "contact_vs_avg_reg": "Contact Over Expected (%)",
                "Swing_pct_reg": "Swing (%)",
                "p_Swing_with_loc_pct_reg": "pSwing (%)",
            }
            df = df.rename(columns=rename_map)
            df = df.sort_values(by="Damage/BBE (%)", ascending=False)
            stats_df = base_stats[
                [col for col in columns if col in base_stats.columns]
            ].rename(columns=rename_map)
            render_table(
                df,
                reverse_cols=HIGHER_IS_WORSE_COLS | {"LA<=0%", "Chase (%)"},
                group_cols=["__season", "__level"],
                stats_df=stats_df,
            )
            download_button(df, "hitters_ar", "hitters_ar_download")


def hitter_splits():
    """Hitters - Splits page"""
    st.title("Hitter Splits")

    if hitter_splits_df.empty:
        st.info("Missing hitter_splits.csv")
        return

    tabs = st.tabs(["vL / vR", "Home / Away", "1H / 2H", "Monthly"])
    split_map = {
        "vL / vR": "vs L/R",
        "Home / Away": "Home/Away",
        "1H / 2H": "1st Half/2nd Half",
        "Monthly": "Monthly",
    }

    for idx, tab_name in enumerate(split_map.keys()):
        with tabs[idx]:
            split_type = split_map[tab_name]
            split_df = hitter_splits_df[
                hitter_splits_df["split_type"] == split_type
            ].copy()
            if split_df.empty:
                available = sorted(
                    hitter_splits_df["split_type"].dropna().unique().tolist()
                )
                st.info(f"No data for {tab_name}. Available split types: {available}")
                continue

            left, right = st.columns([1, 3])
            with left:
                level = st.selectbox(
                    "Select Level",
                    ["All", "MLB", "Triple-A", "Low-A", "Low Minors"],
                    index=1,
                    key=f"hitter_splits_level_{idx}",
                )
                season = st.multiselect(
                    "Select Season",
                    season_options(split_df),
                    default=(
                        [season_options(split_df)[1]]
                        if len(season_options(split_df)) > 1
                        else ["All"]
                    ),
                    key=f"hitter_splits_season_{idx}",
                )
                game_type_group = st.selectbox(
                    "Game Type",
                    game_type_group_options(hitter_splits_df),
                    index=0,
                    key=f"hitter_splits_game_type_group_{idx}",
                )
                min_value = st.number_input(
                    "Minimum Value",
                    min_value=0,
                    max_value=500,
                    value=100,
                    step=1,
                    key=f"hitter_splits_min_value_{idx}",
                )
                value_type = st.selectbox(
                    "Filter By",
                    ["PA", "BBE"],
                    index=1,
                    key=f"hitter_splits_value_type_{idx}",
                )
                split_choice = st.multiselect(
                    "Select Split",
                    ["All"] + sorted(split_df["split"].dropna().unique().tolist()),
                    default=["All"],
                    key=f"hitter_splits_split_{idx}",
                )
                team = st.selectbox(
                    "Select Team",
                    team_options(split_df, "hitting_code"),
                    index=0,
                    key=f"hitter_splits_team_{idx}",
                )
                position = st.multiselect(
                    "Select Position",
                    position_options(split_df),
                    default=["All"],
                    key=f"hitter_splits_position_{idx}",
                    format_func=lambda v: (
                        "All" if v == "All" else POSITION_FILTER_LABELS.get(v, v)
                    ),
                )
                player_options_list, player_name_map = player_id_options(
                    split_df, "batter_mlbid", "hitter_name"
                )
                player = st.multiselect(
                    "Select Player",
                    player_options_list,
                    default=["All"],
                    format_func=lambda v: (
                        "All"
                        if v == "All"
                        else f"{player_name_map.get(v, 'Unknown')} ({int(v)})"
                    ),
                    key=f"hitter_splits_player_{idx}",
                )
            with right:
                if game_type_group != "Regular Season":
                    st.info(GAME_TYPE_GROUP_NOTE.format(game_type_group))
                level_map = {
                    "All": [1, 11, 14, 16],
                    "MLB": [1],
                    "Triple-A": [11],
                    "Low-A": [14],
                    "Low Minors": [16],
                }
                base_stats = split_df.copy()
                base_stats = base_stats.assign(
                    __season=base_stats["season"],
                    __level=base_stats["level_id"],
                )
                df = split_df.copy()
                df = df[df["level_id"].isin(level_map[level])]
                df = filter_by_values(df, "season", season)
                df = filter_by_game_type_group(df, game_type_group)
                df = filter_by_values(df, "split", split_choice)
                df = filter_by_team_token(df, "hitting_code", team)
                df = filter_by_positions(df, position)
                df = filter_by_values(df, "batter_mlbid", player)
                df = df.assign(__season=df["season"], __level=df["level_id"])

                if value_type == "PA":
                    df = numeric_filter(df, "PA", min_value)
                else:
                    df = numeric_filter(df, "bbe", min_value)

                columns = [
                    "hitter_name", "batter_mlbid", "hitting_code", "season", "split",
                    "PA", "bbe", "HR", "damage_rate", "EV90th", "max_EV", "pull_FB_pct",
                    "LA_gte_20", "LA_lte_0", "SEAGER", "selection_skill",
                    "hittable_pitches_taken", "chase", "z_con", "secondary_whiff_pct",
                    "whiffs_vs_95", "contact_vs_avg", "__season", "__level",
                ]
                df = df[[col for col in columns if col in df.columns]].copy()
                rename_map = {
                    "hitter_name": "Name", "batter_mlbid": "Player ID",
                    "hitting_code": "Team", "season": "Season", "split": "Split",
                    "bbe": "BBE", "HR": "HR", "damage_rate": "Damage/BBE (%)",
                    "EV90th": "90th Pctile EV", "max_EV": "Max EV",
                    "pull_FB_pct": "Pulled FB (%)", "LA_gte_20": "LA>=20%",
                    "LA_lte_0": "LA<=0%", "selection_skill": "Selectivity (%)",
                    "hittable_pitches_taken": "Hittable Pitch Take (%)",
                    "chase": "Chase (%)", "z_con": "Z-Contact (%)",
                    "secondary_whiff_pct": "Whiff vs. Secondaries (%)",
                    "whiffs_vs_95": "Whiff vs. 95+ (%)",
                    "contact_vs_avg": "Contact Over Expected (%)",
                }
                df = df.rename(columns=rename_map)
                df = df.sort_values(by="Damage/BBE (%)", ascending=False)
                stats_df = base_stats[
                    [col for col in columns if col in base_stats.columns]
                ].rename(columns=rename_map)
                render_table(
                    df,
                    reverse_cols=HIGHER_IS_WORSE_COLS | {"Chase (%)", "LA<=0%"},
                    group_cols=["__season", "__level"],
                    stats_df=stats_df,
                )
                download_button(
                    df,
                    f"hitter_splits_{idx}",
                    f"hitter_splits_download_{idx}",
                )

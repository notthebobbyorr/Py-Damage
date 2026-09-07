from __future__ import annotations

import pandas as pd
import streamlit as st

from app.filters import download_button, filter_by_game_type_group, game_type_group_options, player_id_options
from app.utils import _hitter_display_map, _pitcher_display_map
from app.viz import render_table


def compare_periods(before, after, id_col, name_col, metric, workload, minimum, keep_missing=False):
    """Match players within a level; require qualifying samples in both periods."""
    keys = [id_col, "level_id"]

    frames = []
    for frame in [before, after]:
        frame = frame.copy()
        for col in [metric, workload]:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.dropna(subset=keys)
        if frame.duplicated(keys).any():
            rows = []
            for identity, group in frame.groupby(keys, observed=True):
                row = dict(zip(keys, identity))
                names = group[name_col].dropna()
                row[name_col] = names.iloc[0] if len(names) else "Unknown"
                row[workload] = group[workload].sum(min_count=1)
                values = group[metric].dropna()
                if len(group) == 1 or values.nunique() == 1:
                    value = values.iloc[0] if len(values) else float("nan")
                elif metric + "_num" in group and metric + "_den" in group:
                    denominator = group[metric + "_den"].sum()
                    value = 100 * group[metric + "_num"].sum() / denominator if denominator > 0 else float("nan")
                elif metric == "EV90th":
                    value = float("nan")
                elif metric in ["max_EV", "max_velo"]:
                    value = values.max()
                else:
                    weight_col = metric + "_n" if metric + "_n" in group else "pitches" if metric in ["stuff", "grade_v13"] else None
                    if weight_col:
                        valid = group[metric].notna() & group[weight_col].gt(0)
                        weights = group.loc[valid, weight_col]
                        value = (group.loc[valid, metric] * weights).sum() / weights.sum() if weights.sum() > 0 else float("nan")
                    else:
                        value = float("nan")
                row[metric] = value
                rows.append(row)
            frame = pd.DataFrame(rows)
        frame = frame.dropna(subset=[workload] if keep_missing else [metric, workload])
        frame = frame[frame[workload] >= minimum]
        for key in keys:
            frame[key] = frame[key].astype(str) if key == "__team_code" else pd.to_numeric(frame[key], errors="raise").astype("int64")
        frames.append(frame[keys + [name_col, metric, workload]])
    result = frames[0].merge(frames[1], on=keys, suffixes=("_before", "_after"), validate="one_to_one")
    result["Change"] = result[f"{metric}_after"] - result[f"{metric}_before"]
    result["Name"] = result[f"{name_col}_after"].astype(str).str.normalize("NFKD").str.encode("ascii", errors="ignore").str.decode("ascii")
    return result[["Name"] + keys + [f"{metric}_before", f"{metric}_after", "Change", f"{workload}_before", f"{workload}_after"]].rename(columns={
        id_col: "Player ID", "level_id": "Level ID", "pitcher_hand": "Hand", f"{metric}_before": "Before", f"{metric}_after": "After",
        f"{workload}_before": f"Before {workload}", f"{workload}_after": f"After {workload}",
    })


def player_profile(before, after, player_id, id_col, name_col, metrics, workload, minimum):
    """One row per available metric; preserve missing measurements as blanks."""
    before = before[before[id_col] == player_id]
    after = after[after[id_col] == player_id]
    if before.empty or after.empty:
        return pd.DataFrame()
    if any(not pd.to_numeric(frame[workload], errors="coerce").sum(min_count=1) >= minimum for frame in [before, after]):
        return pd.DataFrame()
    rows = []
    counts = {workload: workload, "pitches": "Pitches", "bbe": "BBE", "HR": "HR", "GS": "GS", "IP": "IP", "SBO": "SBO", "SB": "SB"}
    for col, label in counts.items():
        if col in before.columns or col in after.columns:
            values = [pd.to_numeric(frame[col], errors="coerce").sum(min_count=1) if col in frame else float("nan") for frame in [before, after]]
            values = [float(value) if pd.notna(value) else float("nan") for value in values]
            rows.append({"Metric": label, "Before": values[0], "After": values[1], "Change": values[1] - values[0]})
    for metric, label in metrics.items():
        result = compare_periods(before, after, id_col, name_col, metric, workload, minimum, keep_missing=True)
        if not result.empty:
            values = result.iloc[0]
            rows.append({"Metric": label, "Before": values["Before"], "After": values["After"], "Change": values["Change"]})
    profile = pd.DataFrame(rows)
    for col in ["Before", "After", "Change"]:
        profile[col] = pd.to_numeric(profile[col], errors="coerce").astype("Float64")
    return profile


def render_player_profile(source, before, after, id_col, name_col, metrics, workload, minimum, prefix, team=False):
    entity = "team" if team else "player"
    periods = source[source["period"].isin([before, after])]
    if team:
        options = sorted(periods[id_col].dropna().unique().tolist())
        names = {value: value for value in options}
    else:
        options, names = player_id_options(periods, id_col, name_col)
        options = [value for value in options if value != "All"]
    if not options:
        st.info("No players available for these periods.")
        return
    player = st.selectbox(entity.title(), options, format_func=lambda value: str(value) if team else f"{names.get(value, 'Unknown')} ({int(value)})", key=prefix + "_player")
    profile = player_profile(source[source["period"] == before], source[source["period"] == after], player, id_col, name_col, metrics, workload, minimum)
    st.caption(f"{names.get(player, 'Unknown')}: {after} minus {before}. Percentage metrics use percentage-point differences. Blank values indicate missing measurements or source rows that cannot be combined reliably. Incomplete periods use available data.")
    if profile.empty:
        st.info(f"This {entity} needs data and the minimum sample in both periods. Lower the minimum or choose other periods.")
        return
    render_table(profile, stats_df=pd.DataFrame())
    download = profile.assign(Player=names.get(player, "Unknown"), **{"Player ID": player, "Before period": before, "After period": after})
    if team:
        download = download.rename(columns={"Player": "Team"}).drop(columns="Player ID")
    download_button(download, prefix + "_profile", prefix + "_profile_download")


def render_changes(season_df, splits_df, kind, team=False):
    st.title(f"{'Team Hitting' if kind == 'Hitter' else 'Team Pitching'} Changes" if team else f"{kind} Changes")
    hitter = kind == "Hitter"
    id_col, name_col, workload = ("batter_mlbid", "hitter_name", "PA") if hitter else ("pitcher_mlbid", "name", "TBF")
    if team:
        team_col = "hitting_code" if hitter else "pitching_code"
        season_df = season_df.assign(__team_code=season_df[team_col], __team_name=season_df[team_col])
        splits_df = splits_df.assign(__team_code=splits_df[team_col], __team_name=splits_df[team_col])
        id_col, name_col = "__team_code", "__team_name"
    prefix = f"{'team_' if team else ''}{kind.lower()}_changes"
    mode = st.selectbox("Comparison", ["Year over year", "Month over month", "1st Half vs 2nd Half"], key=prefix + "_mode")
    source = season_df.copy() if mode == "Year over year" else splits_df[splits_df["split_type"] == ("Monthly" if mode == "Month over month" else "1st Half/2nd Half")].copy()
    if source.empty:
        st.info("No data available for this comparison.")
        return
    level_map = {"MLB": 1, "Triple-A": 11, "Low-A": 14, "Low Minors": 16}
    level = st.selectbox("Level", list(level_map), key=prefix + "_level")
    game_type = st.selectbox("Game Type", game_type_group_options(source), key=prefix + "_gt")
    source = filter_by_game_type_group(source[source["level_id"] == level_map[level]], game_type)
    source["season"] = pd.to_numeric(source["season"], errors="coerce")
    source = source.dropna(subset=["season"])
    source["period"] = source["season"].astype(int).astype(str)
    if mode != "Year over year":
        source["period"] += " / " + source["split"].astype(str)
    months = ["February", "March/April", "May", "June", "July", "August", "September/October"]
    def period_order(period):
        parts = period.split(" / ")
        return (int(parts[0]), (months.index(parts[1]) if parts[1] in months else 0 if parts[1] == "1st Half" else 1) if len(parts) > 1 else 0)
    periods = sorted(source["period"].unique(), key=period_order)
    if len(periods) < 2:
        st.info("At least two periods are required.")
        return
    before_options = periods[:-1]
    if mode == "1st Half vs 2nd Half":
        before_options = [p for p in periods if p.endswith("1st Half") and p.replace("1st Half", "2nd Half") in periods]
    if not before_options:
        st.info("No complete comparison pairs available.")
        return
    before = st.selectbox("Before period", before_options, index=len(before_options)-1, key=prefix + "_before")
    later = [p for p in periods if period_order(p) > period_order(before)]
    if mode == "1st Half vs 2nd Half":
        later = [p for p in later if p == before.split(" / ")[0] + " / 2nd Half" and before.endswith("1st Half")]
    if not later:
        st.info("Choose a first half with an available second half in the same season.")
        return
    after = st.selectbox("After period", later, index=len(later)-1, key=prefix + "_after")
    mapping = _hitter_display_map() if hitter else _pitcher_display_map()
    metrics = {col.removesuffix("_reg"): label for col, label in mapping.items() if col not in [name_col, "season", "bbe", "hitting_code", "pitching_code", "similarity_score"] and col.removesuffix("_reg") in source.columns}
    if not metrics:
        st.info("No comparable metrics available.")
        return
    view = st.radio("View", ["By Stat", "By Team" if team else "By Player"], horizontal=True, key=prefix + "_view")
    minimum = st.number_input(f"Minimum {workload} in each period", min_value=0, value=100, key=prefix + "_min")
    if view != "By Stat":
        profile_metrics = dict(metrics)
        extras = {"Swing_pct": "Swing (%)", "p_Swing_with_loc_pct": "pSwing (%)"} if hitter else {"p_SwStr_pct": "pSwStr (%)", "Swing_pct": "Swing (%)", "p_Swing_pct": "pSwing (%)", "Damage_pct": "Damage/BBE (%)", "p_Damage_pct": "pDamage/BBE (%)"}
        profile_metrics.update({col: label for col, label in extras.items() if col in source.columns})
        render_player_profile(source, before, after, id_col, name_col, profile_metrics, workload, minimum, prefix, team=team)
        return
    metric = st.selectbox("Metric", list(metrics), format_func=metrics.get, key=prefix + "_metric")
    order = st.selectbox("Sort by", ["Largest absolute change", "Largest increase", "Largest decrease"], key=prefix + "_sort")
    try:
        result = compare_periods(source[source["period"] == before], source[source["period"] == after], id_col, name_col, metric, workload, minimum)
    except ValueError as exc:
        st.error(str(exc))
        return
    result = result.sort_values("Change", ascending=order == "Largest decrease", key=lambda s: s.abs() if order == "Largest absolute change" else s)
    st.caption(f"{metrics[metric]}: {after} minus {before}. Percentage metrics use percentage-point differences. Incomplete periods use available data; increases do not always indicate improvement. Entities with missing values or source rows that cannot be combined reliably are excluded.")
    if result.empty:
        st.info("No matching entries qualify in both periods.")
        return
    if team:
        result = result.rename(columns={"Name": "Team"}).drop(columns="Player ID")
    render_table(result, stats_df=pd.DataFrame())
    download_button(result, prefix, prefix + "_download")

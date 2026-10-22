"""Selective loading and display of observed regular-season split intersections."""
import pandas as pd
import streamlit as st
from app.filters import apply_workload_maximum

from app.config import DATA_DIR, LEVEL_LABELS, ABS_GRADIENT_COLS_PITCHERS, ABS_GRADIENT_COLS_PITCH_TYPES, DEFAULT_NO_FORMAT_COLS
from app.data_loader import load_csv
from app.filters import split_workload_controls, numeric_filter, filter_by_values, download_button
from app.hitter_pitch_splits import COLUMNS, HITTING_REVERSE_COLUMNS
from app.viz import render_table
from app.combined_split_config import DIMENSIONS, PITCH_COLUMNS


def render_combined_splits(kind):
    prefix = kind + "_combined"
    team_mode = kind.startswith("team_")
    hitting = kind in {"hitter","team_hitter"}
    available = dict(DIMENSIONS)
    if not hitting:
        available.pop("Pitch Group")
    left, right = st.columns([1, 3])
    with left:
        selected = st.multiselect("Combine splits", list(available),
            default=["Opponent Hand", "Pitch Group" if hitting else "Half"], key=prefix+"_dimensions")
        if len(selected) < 2:
            st.info("Select at least two splits.")
            return
        if {"Month", "Half"}.issubset(selected):
            st.info("Choose Month or Half for the time split.")
            return
        dims = [column for label, column in available.items() if label in selected]
        folder = DATA_DIR / "combined_splits" / kind / "__".join(dims)
        paths = sorted(folder.glob("season_*.level_*.parquet"))
        if not paths:
            st.info("These combined splits have not been generated yet.")
            return
        years = sorted({int(p.name.split('.')[0].split('_')[1]) for p in paths}, reverse=True)
        seasons = st.multiselect("Season", years, default=years[:1], key=prefix+"_season")
        level = st.selectbox("Level", [1,11,14,16], format_func=lambda v: LEVEL_LABELS.get(v,str(v)), key=prefix+"_level")
        chosen = [p for p in paths if int(p.name.split('.')[0].split('_')[1]) in seasons and f".level_{level}." in p.name]
        if not chosen:
            st.info("No data for this season and level.")
            return
        source = pd.concat([load_csv(str(p.relative_to(DATA_DIR))) for p in chosen], ignore_index=True)
        df = source
        for label, column in available.items():
            if column in dims:
                values = sorted(source[column].dropna().unique().tolist())
                selection = st.multiselect(label, ["All", *values], default=["All"], key=prefix+column)
                df = filter_by_values(df,column,selection)
        if team_mode:
            id_col = "hitting_code" if hitting else "pitching_code"
            name_col = id_col
            teams = st.multiselect("Team",["All",*sorted(source[id_col].dropna().unique())],default=["All"],key=prefix+"_team")
            df = filter_by_values(df,id_col,teams)
        else:
            id_col, name_col = ("batter_mlbid","hitter_name") if kind == "hitter" else ("pitcher_mlbid","name")
            names = source.drop_duplicates(id_col).set_index(id_col)[name_col].to_dict()
            players = st.multiselect("Player", ["All", *sorted(names, key=lambda v: str(names[v]))], default=["All"],
                format_func=lambda v: "All" if v == "All" else f"{names[v]} ({int(v)})", key=prefix+"_player")
            df = filter_by_values(df,id_col,players)
            if "team" in source:
                from app.filters import team_options, filter_by_team_token
                team = st.selectbox("Team",team_options(source,"team"),key=prefix+"_team")
                df = filter_by_team_token(df,"team",team)
        if "pitch_tag" in source:
            types = st.multiselect("Pitch Type", ["All", *sorted(source.pitch_tag.dropna().unique())], default=["All"], key=prefix+"_pitch")
            df = filter_by_values(df,"pitch_tag",types)
        column, minimum = split_workload_controls(source,prefix,pitching=not hitting)
        df = numeric_filter(df,column,minimum)
        df = apply_workload_maximum(df,prefix)
    with right:
        st.caption("Regular season only. Each row is the intersection of the selected splits. Usage includes all pitches in that same slice.")
        if kind == "pitch":
            st.caption("Execution Grade uses available 2026 scored pitches; blank when fewer than five scored pitches are available.")
        mapping = {id_col:"Player ID",name_col:"Name","season":"Season","team":"Team","pitcher_hand":"Pitcher Hand","pitch_tag":"Pitch Type",
                   **{v:k for k,v in available.items()}, **(COLUMNS if hitting else PITCH_COLUMNS)}
        if team_mode:
            mapping = {id_col:"Team",**{key:value for key,value in mapping.items() if key != id_col}}
        mapping.update({"PA":"PA"})
        if kind == "pitch" and "grade_v13" not in df:
            df = df.assign(grade_v13=float("nan"))
        df = df.assign(__level=df.level_id, __season=df.season)
        keep = [c for c in mapping if c in df] + ["__level","__season"]
        display = df[keep].rename(columns=mapping).sort_values("#",ascending=False)
        stats = source.assign(__level=source.level_id, __season=source.season)
        stats = stats[[c for c in mapping if c in stats] + ["__level", "__season"]].rename(columns=mapping)
        reverse = HITTING_REVERSE_COLUMNS if hitting else {"Ball (%)", "Z-Contact (%)", "VAA", "FA VAA", "Damage/BBE (%)", "pDamage/BBE (%)"}
        display = render_table(display,default_sort_col="#",workload_kind="hitter" if hitting else "pitcher",show_maximum_filters=False,
            reverse_cols=reverse, group_cols=["__season", "__level"], stats_df=stats,
            no_format_cols=DEFAULT_NO_FORMAT_COLS | {"%"} if hitting else None,
            abs_cols=(ABS_GRADIENT_COLS_PITCH_TYPES if kind == "pitch" else ABS_GRADIENT_COLS_PITCHERS) if not hitting else None,
            hide_cols={"TBF","GS","IP"} if kind == "pitch" else {"HR"} if kind == "pitcher" else None,
            damage_after_la=(kind == "pitcher"), pitch_outcomes_after_csw=(kind == "pitch"))
        download_button(display,"combined_splits",prefix+"_download")

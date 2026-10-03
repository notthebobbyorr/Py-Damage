"""Saved official totals and the hitter-only expandable table pilot."""
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v2 as components

from app.config import DATA_DIR

DETAIL_COLUMNS = ["GP", "PA", "AB", "H", "HR", "AVG/OBP/SLG", "OPS", "wRC+", "BABIP", "K%", "BB%", "R", "RBI", "SB", "CS"]


@st.cache_resource
def load_official_hitting():
    path = Path(DATA_DIR) / "official_hitting.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path).set_index(["player_id", "season", "level_id"], verify_integrity=True)


def detail_values(row):
    def number(value, decimals=0):
        return "—" if pd.isna(value) else f"{value:.{decimals}f}"

    slash = "/".join(number(row[col], 3).removeprefix("0") for col in ["AVG", "OBP", "SLG"])
    values = []
    for col in DETAIL_COLUMNS:
        if col == "AVG/OBP/SLG":
            values.append(slash)
        elif col == "wRC+":
            values.append(number(row.get(col, float("nan"))))
        elif col in {"OPS", "BABIP"}:
            values.append(number(row[col], 3).removeprefix("0"))
        elif col.endswith("%"):
            values.append("—" if pd.isna(row[col]) else number(row[col], 1) + "%")
        else:
            values.append(number(row[col]))
    return values


_component = components.component(
    "official_hitter_table",
    html='<div class="table-toolbar"><details class="column-picker"><summary>Columns</summary><div class="column-options"></div></details><button type="button" class="fullscreen-toggle">Fullscreen</button></div><div class="table-scroll"></div><dialog class="table-fullscreen" aria-label="Fullscreen stats table"></dialog>',
    css="""
    .table-toolbar {display:flex; justify-content:flex-end; gap:8px; margin-bottom:4px; position:relative;}
    .column-picker summary {cursor:pointer; padding:5px 10px; border:1px solid #d5d9df; border-radius:5px;}
    .column-options {position:absolute; right:0; top:100%; z-index:5; width:310px; max-height:50vh; overflow:auto; padding:10px; border:1px solid #d5d9df; border-radius:5px; background:var(--st-background-color); color:var(--st-text-color); box-shadow:0 4px 12px #0003;}
    .column-option {display:flex; align-items:center; gap:6px; padding:4px 0;}
    .column-option label {flex:1; font:13px sans-serif;}
    .column-option button {cursor:pointer; background:var(--st-background-color); color:var(--st-text-color); border:1px solid #d5d9df; border-radius:3px;}
    .fullscreen-toggle {cursor:pointer; padding:5px 10px; border:1px solid #d5d9df; border-radius:5px; background:var(--st-background-color); color:var(--st-text-color);}
    .table-fullscreen {box-sizing:border-box; width:calc(100vw - 24px); height:calc(100vh - 24px); max-width:none; max-height:none; padding:12px; border:1px solid #d5d9df; border-radius:5px; background:var(--st-background-color); color:var(--st-text-color);}
    .table-fullscreen[open] {display:flex; flex-direction:column;}
    .table-fullscreen::backdrop {background:rgba(0,0,0,0.45);}
    .table-fullscreen .table-scroll {flex:1; min-height:0; max-height:none;}
    .table-scroll {overflow:auto; max-height:620px; border:1px solid #d5d9df; border-radius:5px;}
    table {border-collapse:collapse; font:13px sans-serif; color:var(--st-text-color); width:max-content; min-width:100%;}
    th, td {padding:8px 10px; border-bottom:1px solid #d5d9df; text-align:right; white-space:nowrap;}
    thead th {position:sticky; top:0; z-index:1; background:var(--st-secondary-background-color); cursor:pointer;}
    tbody tr.player {cursor:pointer;}
    tbody tr.player:hover {outline:1px solid #8793a4; outline-offset:-1px;}
    tbody tr.player:focus {outline:2px solid var(--st-primary-color); outline-offset:-2px;}
    th:first-child, td:first-child {text-align:left;}
    tr.detail td {background:var(--st-background-color); color:var(--st-text-color); font-weight:normal;}
    tr.detail table {width:auto; min-width:0;}
    tr.detail th, tr.detail td {font-weight:normal; border:0; padding:5px 10px;}
    """,
    js=Path(__file__).with_name("official_hitting.js").read_text(encoding="utf-8"),
)


def render_official_table(display, full, key, pitching=False, team=False):
    if team:
        from app.official_teams import load_official_teams, detail_columns, detail_values as team_values
        official = load_official_teams(pitching)
        columns = detail_columns(pitching)
        format_values = lambda row: team_values(row, pitching)
    elif pitching:
        from app.official_pitching import load_official_pitching, detail_values as format_values, DETAIL_COLUMNS as columns
        official = load_official_pitching()
    else:
        official = load_official_hitting()
        format_values, columns = detail_values, DETAIL_COLUMNS
    details = []
    for _, row in full.iterrows():
        entity = str(row["Team"]) if team else int(row["Player ID"])
        if team:
            entity = {"OAK": "ATH", "AZ": "ARI"}.get(entity, entity)
        identity = (entity, int(row["Season"]), int(row.get("__official_level", row["__level"])))
        details.append(format_values(official.loc[identity]) if row.get("__official_game_type", "Regular Season") == "Regular Season" and identity in official.index else None)
    styler = display if hasattr(display, "hide") else display.style
    # Escape all source text before putting the existing table styles into HTML.
    html = styler.format(escape="html", subset=styler.data.select_dtypes(exclude="number").columns).format_index(escape="html", axis=1).hide(axis="index").to_html()
    st.caption(("Click a team row to show official regular-season team totals for this level." if team else "Click a player row to show official regular-season totals for this level across all teams.") + " Click again to collapse." + ((" Hld = holds; HA = hits allowed." + ("" if team else " Inh. Runners % = inherited runners who scored.")) if pitching else ""))
    _component(data={"html": html, "details": details, "columns": columns}, key=key)

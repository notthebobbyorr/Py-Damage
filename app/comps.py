"""Comparison presets and source-metric preparation."""
import pandas as pd
import streamlit as st
from app.comparison_metrics import MAIN_METRICS, comparison_metric

PRESETS = {
    "hitter": {
        "Overall": "pull_FB_pct_reg EV90th_reg damage_rate_reg takeoff_rate_reg contact_vs_avg_reg z_con_reg chase_reg SEAGER_reg LA_lte_0_reg LA_gte_20_reg",
        "Over the plate": "hittable_pitches_taken_reg selection_skill_reg SEAGER_reg Swing_pct_reg contact_vs_avg_reg whiffs_vs_95_reg secondary_whiff_pct_reg z_con_reg chase_reg",
        "Batted Ball": "EV90th_reg damage_rate_reg LA_lte_0_reg LA_gte_20_reg pull_FB_pct_reg max_EV_reg",
        "Bat Path": "swing_length_reg fast_swing_pct bat_speed_reg intercept_y_inches intercept_x_inches attack_direction attack_angle_reg swing_path_tilt_reg",
    },
    "pitcher": {
        "Overall": "stuff grade_v13 fastball_velo_reg fastball_vaa_reg FA_spin_eff_reg FA_pct_reg BB_rpm_reg SwStr_reg Ball_pct_reg Z_Contact_reg Chase_reg LA_lte_0_reg inf_arm_angle",
        "Traits & Usage": "fastball_velo_reg fastball_vaa_reg FA_spin_eff_reg FA_pct_reg BB_pct OFF_pct BB_rpm_reg BB_velo OFF_velo rel_z_reg rel_x_reg ext_reg inf_arm_angle",
        "Outcomes": "stuff grade_v13 SwStr_reg Ball_pct_reg Z_Contact_reg Chase_reg LD_pct_reg LA_lte_0_reg LA_gte_20_reg takeoff_rate_reg CSW_reg Zone_reg",
    },
    "pitch": {
        "Shapes": "velo vbreak hbreak rel_z rel_x ext rpm spin_efficiency",
        "Angles": "velo vaa haa z_angle_release x_angle_release inf_arm_angle spin_efficiency rpm",
        "Outcomes": "SwStr Ball_pct Zone Z_Contact LA_lte_0 Chase CSW",
    },
}


def prepare_comparison_metrics(df, kind, equivalent=False):
    df = df.copy()
    # Preserve the existing explicitly untranslated grade/trait metrics.
    if equivalent:
        columns = MAIN_METRICS[kind] + [c for value in PRESETS[kind].values() for c in value.split()]
        for col in columns:
            translated = comparison_metric(kind, col, True)
            if col in df and translated != col and translated not in df and not col.endswith("_reg"):
                df[translated] = df[col]
    return df


def comparison_columns(kind, df, display_map, defaults, key, equivalent=False, available=()):
    resolve = lambda column: comparison_metric(kind, column, equivalent)
    mapping = {name: [resolve(col) for col in value.split()] for name, value in PRESETS[kind].items()}
    labels = dict(display_map)
    denominator = {"hitter": "PA", "pitcher": "TBF", "pitch": "pitches"}[kind]
    labels.update({"HR_rate": f"HR/{denominator}%", "SB_rate": "SB/PA%", "Zone": "Zone%", "Swing_pct_reg": "Swing (%)"})
    for col, label in list(labels.items()):
        labels[col + "_mlb_eq"] = label
    defaults = list(dict.fromkeys(resolve(c) for c in defaults))
    options = list(dict.fromkeys([*defaults, *(resolve(c) for c in available),
        *(resolve(c) for c in MAIN_METRICS[kind]), *[c for values in mapping.values() for c in values]]))
    counts = {"HR", "SB", "SBO", "PA", "TBF", "IP", "GS", "bbe", "pitches", "season"}
    options = [c for c in options if c.removesuffix("_mlb_eq") not in counts and c in df and pd.api.types.is_numeric_dtype(df[c])]
    preset_key = key + "_preset"
    def select_preset():
        name = st.session_state[preset_key]
        if name != "Custom":
            st.session_state[key] = [c for c in mapping[name] if c in options]
    def customize():
        st.session_state[preset_key] = "Custom"
    preset = st.selectbox("Comparison preset", ["Custom", *mapping], key=preset_key, on_change=select_preset)
    if key in st.session_state:
        st.session_state[key] = list(dict.fromkeys(resolve(c) for c in st.session_state[key] if resolve(c) in options))
    else:
        st.session_state[key] = [c for c in defaults if c in options]
    if preset != "Custom":
        missing = [labels.get(c, c) for c in mapping[preset] if c not in options or not df[c].notna().any()]
        if missing:
            st.warning("Unavailable preset metrics: " + ", ".join(missing))
    selected = st.multiselect("Similarity Score Columns", options, key=key, format_func=lambda c: labels.get(c,c), on_change=customize)
    return selected, labels


def translate_fastball_vaa(pitches, pitchers, coefficients):
    """Apply existing pitcher fastball VAA affine translation to fastball pitch rows."""
    out = pitches.copy()
    if "vaa" not in out:
        return out
    base = pitchers.groupby(["pitcher_mlbid", "season", "level_id"], observed=True)["fastball_vaa_reg"].mean().reset_index()
    moments = base.groupby(["season", "level_id"], observed=True)["fastball_vaa_reg"].agg(mean="mean", std=lambda x: x.std(ddof=0))
    coeff = coefficients[(coefficients.metric == "fastball_vaa_reg") & (coefficients.dst_level == 1)].drop_duplicates("src_level", keep="last").set_index("src_level")
    fastballs = out.pitch_group.eq("FA") & out.level_id.ne(1)
    for (season, level), rows in out[fastballs].groupby(["season", "level_id"], observed=True):
        if (season,level) not in moments.index or (season,1) not in moments.index or level not in coeff.index:
            out.loc[rows.index,"vaa"] = float("nan")
            continue
        src, dst = moments.loc[(season,level)], moments.loc[(season,1)]
        c = coeff.loc[level]
        out.loc[rows.index,"vaa"] = dst['mean'] + (c.a + c.b * (rows.vaa-src['mean']) / src['std']) * dst['std'] if src['std'] > 0 else float("nan")
    return out

"""Main-page skill metrics and their intended Comps sources (counts excluded)."""
BAT_METRICS = {
    "bat_speed", "fast_swing_pct", "swing_length", "swing_path_tilt", "attack_angle",
    "attack_direction", "intercept_x_inches", "intercept_y_inches", "miss_distance",
}
OBSERVED_METRICS = {
    "hitter": BAT_METRICS,
    "pitcher": {"miss_distance", "FA_pct", "BB_pct", "OFF_pct"},
    "pitch": set(),
}
MAIN_METRICS = {
    "hitter": (
        "damage_rate_reg EV90th_reg max_EV_reg pull_FB_pct_reg LA_gte_20_reg LA_lte_0_reg "
        "SEAGER_reg selection_skill_reg hittable_pitches_taken_reg chase_reg z_con_reg "
        "secondary_whiff_pct_reg whiffs_vs_95_reg contact_vs_avg_reg Swing_pct_reg "
        "p_Swing_with_loc_pct_reg whiff_pct_reg SwStr_reg takeoff_rate_reg "
        "bat_speed fast_swing_pct swing_length swing_path_tilt attack_angle attack_direction "
        "intercept_x_inches intercept_y_inches miss_distance"
    ).split(),
    "pitcher": (
        "stuff grade_v13 fastball_velo_reg max_velo_reg fastball_vaa_reg FA_pct BB_pct OFF_pct "
        "BB_rpm_reg BB_velo OFF_velo SwStr_reg p_SwStr_pct_reg Swing_pct_reg p_Swing_pct_reg "
        "Damage_pct_reg p_Damage_pct_reg Zone_reg Ball_pct_reg Z_Contact_reg Chase_reg CSW_reg "
        "LA_lte_0_reg rel_z_reg rel_x_reg ext_reg inf_arm_angle takeoff_rate_reg "
        "whiff_pct_reg miss_distance"
    ).split(),
    "pitch": (
        "pct stuff grade_v13 velo max_velo vaa haa z_angle_release x_angle_release vbreak hbreak "
        "rpm spin_efficiency SwStr p_SwStr_pct Swing_pct p_Swing_pct Damage_pct p_Damage_pct "
        "LA_lte_0 Z_Contact Ball_pct Zone Chase CSW rel_z rel_x ext miss_distance whiff_pct"
    ).split(),
}


def comparison_metric(kind, column, equivalent=False):
    """Observed exceptions stay observed in either mode; modeled metrics never fall back."""
    base = column.removesuffix("_mlb_eq")
    raw = base.removesuffix("_reg")
    if raw in OBSERVED_METRICS[kind]:
        return raw
    return base + "_mlb_eq" if equivalent else base

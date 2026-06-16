from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "output"

# ---------------------------------------------------------------------------
# UI / display
# ---------------------------------------------------------------------------
DEFAULT_NO_FORMAT_COLS = {"Season", "PA", "BBE", "TBF", "IP", "GS", "Age", "HR"}

# Columns where higher values are worse (red=high, green=low) — inverted color scale
HIGHER_IS_WORSE_COLS = {
    "Hittable Pitch Take (%)",
    "Whiff vs. Secondaries (%)",
    "Whiff vs. 95+ (%)",
    "Ball (%)",
}

ABS_GRADIENT_COLS_PITCHERS = {"Horizontal Release (ft.)"}
ABS_GRADIENT_COLS_PITCH_TYPES = {"HAA", "HB (in.)"}

PREVIEW_ROWS = 5

# ---------------------------------------------------------------------------
# Stripe / billing
# ---------------------------------------------------------------------------
ANNUAL_PAYMENT_LINK = "https://buy.stripe.com/6oU14p7OEgrCfTsbyQ6J202"
MONTHLY_PAYMENT_LINK = "https://buy.stripe.com/aFaaEZ0mc6R2cHg5as6J204"

# ---------------------------------------------------------------------------
# Level labels
# ---------------------------------------------------------------------------
LEVEL_LABELS = {1: "MLB", 11: "Triple-A", 14: "Low-A", 16: "Low Minors"}

# ---------------------------------------------------------------------------
# Position filters
# ---------------------------------------------------------------------------
POSITION_FILTER_COLS = ["UT", "C", "X1B", "X2B", "X3B", "SS", "OF", "P"]
POSITION_FILTER_LABELS = {
    "UT": "Utility",
    "C": "Catcher",
    "X1B": "1B",
    "X2B": "2B",
    "X3B": "3B",
    "SS": "SS",
    "OF": "OF",
    "P": "Pitcher",
}
POSITION_COUNT_THRESHOLD = 20

# ---------------------------------------------------------------------------
# Game type groups
# ---------------------------------------------------------------------------
GAME_TYPE_GROUP_OPTIONS = ["Regular Season", "Spring Training", "Postseason"]
GAME_TYPE_GROUP_NOTE = (
    "**Note:** Displaying {} data. Stats may not be directly comparable to Regular Season benchmarks."
)

# ---------------------------------------------------------------------------
# Hitter feature / config constants
# ---------------------------------------------------------------------------
HITTER_COMPS_BASE_FEATURE_COLS = [
    "damage_rate_reg",
    "EV90th_reg",
    "pull_FB_pct_reg",
    "LA_gte_20_reg",
    "LA_lte_0_reg",
    "SEAGER_reg",
    "selection_skill_reg",
    "hittable_pitches_taken_reg",
    "chase_reg",
    "z_con_reg",
    "secondary_whiff_pct_reg",
    "whiffs_vs_95_reg",
    "contact_vs_avg_reg",
]
HITTER_COMPS_EXTRA_FEATURE_COLS = [
    "LD_pct_reg",
    "bat_speed_reg",
    "fast_swing_pct",
    "swing_length_reg",
    "attack_angle_reg",
    "attack_direction",
    "intercept_x_inches",
    "intercept_y_inches",
    "swing_path_tilt_reg",
    "max_EV_reg",
    "takeoff_rate_reg",
]
HITTER_HIGHER_IS_WORSE_METRICS = {
    "hittable_pitches_taken_reg",
    "secondary_whiff_pct_reg",
    "whiffs_vs_95_reg",
    "chase_reg",
    "LA_lte_0_reg",
    "swing_length_reg",
    "attack_angle_reg",
    "swing_path_tilt_reg",
}
HITTER_MLB_DIRECTION_MAP = {
    "damage_rate_reg": "down",
    "pull_FB_pct_reg": "down",
    "LA_gte_20_reg": "down",
    "LA_lte_0_reg": "up",
    "SEAGER_reg": "down",
    "selection_skill_reg": "down",
    "hittable_pitches_taken_reg": "up",
    "chase_reg": "up",
    "z_con_reg": "down",
    "secondary_whiff_pct_reg": "up",
    "whiffs_vs_95_reg": "up",
    "contact_vs_avg_reg": "down",
}
HITTER_MLB_MIN_SHIFT_SCALE = 0.75
HITTER_MLB_MIN_SHIFT_SCALE_OVERRIDES = {
    "LA_gte_20_reg": 1.25,
    "LA_lte_0_reg": 1.5,
}
HITTER_MLB_MIN_SHIFT_FLOOR = {
    "LA_gte_20_reg": 2.0,
    "LA_lte_0_reg": 2.0,
}

# ---------------------------------------------------------------------------
# Pitcher feature / config constants
# ---------------------------------------------------------------------------
PITCHER_COMPS_BASE_FEATURE_COLS = [
    "stuff",
    "grade_v13",
    "fastball_velo_reg",
    "fastball_vaa_reg",
    "FA_pct_reg",
    "BB_rpm_reg",
    "SwStr_reg",
    "Ball_pct_reg",
    "Z_Contact_reg",
    "Chase_reg",
    "LA_lte_0_reg",
    "rel_z_reg",
    "rel_x_reg",
    "ext_reg",
    "inf_arm_angle",
]
PITCHER_COMPS_EXTRA_FEATURE_COLS = [
    "Zone_reg",
    "CSW_reg",
    "FA_spin_eff_reg",
    "LD_pct_reg",
    "LA_gte_20_reg",
    "takeoff_rate_reg",
]
PITCHER_MLB_PASS_THROUGH_COLS = {
    "stuff",
    "stuff_raw_reg",
    "fastball_velo_reg",
    "rel_z_reg",
    "rel_x_reg",
    "ext_reg",
    "inf_arm_angle",
}
PITCHER_MLB_HIGHER_IS_WORSE_METRICS = {
    "Ball_pct_reg",
    "Z_Contact_reg",
    "LD_pct_reg",
    "LA_gte_20_reg",
    "takeoff_rate_reg",
}
PITCHER_MLB_DIRECTION_MAP = {
    "SwStr_reg": "down",
    "Chase_reg": "down",
    "LA_lte_0_reg": "down",
    "CSW_reg": "down",
    "Ball_pct_reg": "up",
    "Z_Contact_reg": "up",
    "LD_pct_reg": "up",
    "LA_gte_20_reg": "up",
}
PITCHER_MLB_MIN_SHIFT_SCALE = 0.75
PITCHER_REVERSE_DISPLAY_COLS = {
    "Ball (%)",
    "FA VAA",
    "Z-Contact (%)",
    "0<LA<20 (%)",
    "LA>=20 (%)",
    "Inferred Arm Angle",
    "Takeoff% Against",
}

# ---------------------------------------------------------------------------
# Pitch (pitch_tag-level) comps feature columns
# Raw (non-regressed) per user direction — pitch shapes regress quickly.
# ---------------------------------------------------------------------------
PITCH_COMPS_BASE_FEATURE_COLS = [
    "velo",
    "vbreak",
    "hbreak",
    "rel_z",
    "ext",
    "inf_arm_angle",
    
]
PITCH_COMPS_EXTRA_FEATURE_COLS = [
    "max_velo",
    "vaa",
    "haa",
    "rel_x",
    "z_angle_release",
    "x_angle_release",
    "rpm",
    "spin_efficiency",
    "stuff",
    "grade_v13",
    "Z_Contact",
    "SwStr",
    "p_SwStr_pct",
    "Zone",
    "Chase",
    "Ball_pct",
    "CSW",
    "p_Damage_pct",
    "LA_lte_0",
    "HR",
]
PITCH_REVERSE_DISPLAY_COLS = {
    "Ball (%)",
    "VAA",
    "Z-Contact (%)",
}

# ---------------------------------------------------------------------------
# Radar comparison templates
# Each spoke entry: (raw_col, display_label, higher_is_worse?)
# `higher_is_worse=True` inverts the percentile spoke so larger area = better.
# Column order within each template mirrors the order on the Individual Stats
# / Auto-Regressed pages. Dict insertion order = dropdown order; the first
# template under each player type is the default.
# ---------------------------------------------------------------------------
RADAR_TEMPLATES: dict[str, dict[str, dict[str, list[tuple[str, str, bool]]]]] = {
    "Hitter": {
        "Overall (default)": {
            "primary": [
                ("HR", "HR", False),
                ("damage_rate", "Damage/BBE", False),
                ("EV90th", "90th Pctile EV", False),
                ("pull_FB_pct", "Pulled FB", False),
                ("LA_gte_20", "LA>=20", False),
                ("LA_lte_0", "LA<=0", True),
                ("SEAGER", "SEAGER", False),
                ("chase", "Chase", True),
                ("z_con", "Z-Contact", False),
                ("contact_vs_avg", "Contact Over Exp", False),
                ("SB", "SB", False),
                ("takeoff_rate", "Takeoff%", False),
            ],
            "ar": [
                ("HR", "HR", False),
                ("damage_rate_reg", "Damage/BBE", False),
                ("EV90th_reg", "90th Pctile EV", False),
                ("pull_FB_pct_reg", "Pulled FB", False),
                ("LA_gte_20_reg", "LA>=20", False),
                ("LA_lte_0_reg", "LA<=0", True),
                ("SEAGER_reg", "SEAGER", False),
                ("chase_reg", "Chase", True),
                ("z_con_reg", "Z-Contact", False),
                ("contact_vs_avg_reg", "Contact Over Exp", False),
                ("SB", "SB", False),
                ("takeoff_rate_reg", "Takeoff%", False),
            ],
        },
        "Over the Plate": {
            "primary": [
                ("SEAGER", "SEAGER", False),
                ("selection_skill", "Selectivity", False),
                ("hittable_pitches_taken", "Hittable Take", True),
                ("chase", "Chase", True),
                ("z_con", "Z-Contact", False),
                ("secondary_whiff_pct", "Whiff vs Sec", True),
                ("whiffs_vs_95", "Whiff vs 95+", True),
                ("contact_vs_avg", "Contact Over Exp", False),
                ("Swing_pct", "Swing", False),
            ],
            "ar": [
                ("SEAGER_reg", "SEAGER", False),
                ("selection_skill_reg", "Selectivity", False),
                ("hittable_pitches_taken_reg", "Hittable Take", True),
                ("chase_reg", "Chase", True),
                ("z_con_reg", "Z-Contact", False),
                ("secondary_whiff_pct_reg", "Whiff vs Sec", True),
                ("whiffs_vs_95_reg", "Whiff vs 95+", True),
                ("contact_vs_avg_reg", "Contact Over Exp", False),
                ("Swing_pct_reg", "Swing", False),
            ],
        },
        "Batted Ball": {
            "primary": [
                ("HR", "HR", False),
                ("damage_rate", "Damage/BBE", False),
                ("EV90th", "90th Pctile EV", False),
                ("max_EV", "Max EV", False),
                ("pull_FB_pct", "Pulled FB", False),
                ("LA_gte_20", "LA>=20", False),
                ("LA_lte_0", "LA<=0", True),
            ],
            "ar": [
                ("HR", "HR", False),
                ("damage_rate_reg", "Damage/BBE", False),
                ("EV90th_reg", "90th Pctile EV", False),
                ("max_EV_reg", "Max EV", False),
                ("pull_FB_pct_reg", "Pulled FB", False),
                ("LA_gte_20_reg", "LA>=20", False),
                ("LA_lte_0_reg", "LA<=0", True),
            ],
        },
        "Bat Path": {
            "primary": [
                ("bat_speed", "Avg Swing Speed", False),
                ("fast_swing_pct", "Fast Swing", False),
                ("swing_length", "Swing Length", True),
                ("swing_path_tilt", "VBA", True),
                ("attack_angle", "Attack Angle", True),
                ("attack_direction", "Attack Direction", False),
                ("intercept_x_inches", "Intercept X", False),
                ("intercept_y_inches", "Intercept Y", False),
            ],
            "ar": [
                ("bat_speed_reg", "Avg Swing Speed", False),
                ("fast_swing_pct", "Fast Swing", False),
                ("swing_length_reg", "Swing Length", True),
                ("swing_path_tilt_reg", "VBA", True),
                ("attack_angle_reg", "Attack Angle", True),
                ("attack_direction", "Attack Direction", False),
                ("intercept_x_inches", "Intercept X", False),
                ("intercept_y_inches", "Intercept Y", False),
            ],
        },
    },
    "Pitcher": {
        "Overall (default)": {
            "primary": [
                ("HR", "HR", True),
                ("stuff", "Pitch Grade", False),
                ("grade_v13", "Execution Grade", False),
                ("SwStr", "SwStr", False),
                ("Damage_pct", "Damage/BBE", True),
                ("Ball_pct", "Ball", True),
                ("CSW", "CSW", False),
                ("LA_lte_0", "LA<=0", False),
                ("SB", "SB", True),
                ("takeoff_rate", "Takeoff% Against", True),
            ],
            "ar": [
                ("HR", "HR", True),
                ("stuff", "Pitch Grade", False),
                ("grade_v13", "Execution Grade", False),
                ("SwStr_reg", "SwStr", False),
                ("Damage_pct_reg", "Damage/BBE", True),
                ("Ball_pct_reg", "Ball", True),
                ("CSW_reg", "CSW", False),
                ("LA_lte_0_reg", "LA<=0", False),
                ("SB", "SB", True),
                ("takeoff_rate_reg", "Takeoff% Against", True),
            ],
        },
        "Stuff / Traits": {
            "primary": [
                ("stuff", "Pitch Grade", False),
                ("grade_v13", "Execution Grade", False),
                ("fastball_velo", "FA Velo", False),
                ("fastball_vaa", "FA VAA", True),
                ("FA_pct", "FA Usage", False),
                ("BB_rpm", "BB Spin", False),
                ("rel_z", "Vertical Release", False),
                ("rel_x", "Horizontal Release", False),
                ("ext", "Extension", False),
                ("inf_arm_angle", "Arm Angle", False),
            ],
            "ar": [
                ("stuff", "Pitch Grade", False),
                ("grade_v13", "Execution Grade", False),
                ("fastball_velo_reg", "FA Velo", False),
                ("fastball_vaa_reg", "FA VAA", True),
                ("FA_pct_reg", "FA Usage", False),
                ("BB_rpm_reg", "BB Spin", False),
                ("rel_z_reg", "Vertical Release", False),
                ("rel_x_reg", "Horizontal Release", False),
                ("ext_reg", "Extension", False),
                ("inf_arm_angle", "Arm Angle", False),
            ],
        },
        "Over the Plate": {
            "primary": [
                ("SwStr", "SwStr", False),
                ("Swing_pct", "Swing", False),
                ("Zone", "Zone", False),
                ("Ball_pct", "Ball", True),
                ("Z_Contact", "Z-Contact", True),
                ("Chase", "Chase", False),
                ("CSW", "CSW", False),
            ],
            "ar": [
                ("SwStr_reg", "SwStr", False),
                ("Swing_pct", "Swing", False),
                ("Zone", "Zone", False),
                ("Ball_pct_reg", "Ball", True),
                ("Z_Contact_reg", "Z-Contact", True),
                ("Chase_reg", "Chase", False),
                ("CSW_reg", "CSW", False),
            ],
        },
    },
}

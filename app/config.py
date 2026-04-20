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
    "swing_length_reg",
    "attack_angle_reg",
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
]
PITCHER_COMPS_EXTRA_FEATURE_COLS = [
    "Zone_reg",
    "CSW_reg",
    "loc_adj_vaa_reg",
    "FA_spin_eff_reg",
    "LD_pct_reg",
    "LA_gte_20_reg",
    "arm_angle_reg",
    "takeoff_rate_reg",
]
PITCHER_MLB_PASS_THROUGH_COLS = {
    "stuff",
    "stuff_raw_reg",
    "fastball_velo_reg",
    "rel_z_reg",
    "rel_x_reg",
    "ext_reg",
    "arm_angle_reg",
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
    "Arm Angle",
    "Takeoff% Against",
}

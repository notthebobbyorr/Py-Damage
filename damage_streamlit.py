from __future__ import annotations

from pathlib import Path
import re
import sys
from urllib.parse import urlparse

import pandas as pd
import streamlit as st
import numpy as np
from matplotlib import colors
import plotly.express as px
from st_paywall import add_auth

DATA_DIR = Path(__file__).resolve().parent
_TABLE_COUNTER = 0
DEFAULT_NO_FORMAT_COLS = {"Season", "PA", "BBE", "TBF", "IP", "GS", "Age"}
# Columns where higher values are worse (red=high, green=low) - inverted color scale
HIGHER_IS_WORSE_COLS = {
    "Hittable Pitch Take (%)",
    "Whiff vs. Secondaries (%)",
    "Whiff vs. 95+ (%)",
    "Ball (%)",
}
ABS_GRADIENT_COLS_PITCHERS = {"Horizontal Release (ft.)"}
ABS_GRADIENT_COLS_PITCH_TYPES = {"HAA", "HB (in.)"}
ANNUAL_PAYMENT_LINK = "https://buy.stripe.com/6oU14p7OEgrCfTsbyQ6J202"
MONTHLY_PAYMENT_LINK = "https://buy.stripe.com/aFaaEZ0mc6R2cHg5as6J204"
PREVIEW_ROWS = 5
LEVEL_LABELS = {1: "MLB", 11: "Triple-A", 14: "Low-A", 16: "Low Minors"}
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
}


def _get_stripe_api_key() -> str | None:
    try:
        testing_mode = bool(st.secrets.get("testing_mode", False))
    except Exception:
        testing_mode = False
    if testing_mode:
        return st.secrets.get("stripe_api_key_test") or st.secrets.get("stripe_api_key")
    return st.secrets.get("stripe_api_key")


def _infer_return_url() -> str | None:
    return_url = st.secrets.get("billing_portal_return_url") or st.secrets.get(
        "app_url"
    )
    if return_url:
        return return_url
    auth = st.secrets.get("auth", {})
    redirect_uri = auth.get("redirect_uri")
    if not redirect_uri:
        return None
    parsed = urlparse(redirect_uri)
    if not parsed.scheme or not parsed.netloc:
        return None
    return f"{parsed.scheme}://{parsed.netloc}"


def _get_user_email() -> str | None:
    try:
        return st.user.email
    except Exception:
        return None


@st.cache_data(ttl=86400, max_entries=5)
def _create_billing_portal_url(email: str) -> str | None:
    api_key = _get_stripe_api_key()
    if not api_key:
        return None
    try:
        import stripe
    except Exception:
        return None
    return_url = _infer_return_url()
    if not return_url:
        return None
    stripe.api_key = api_key
    try:
        customers = stripe.Customer.list(email=email, limit=1)
        if not customers.data:
            return None
        session = stripe.billing_portal.Session.create(
            customer=customers.data[0].id,
            return_url=return_url,
        )
        return session.url
    except Exception:
        return None


def _run_streamlit_app() -> None:
    import streamlit.web.cli as stcli

    sys.argv = ["streamlit", "run", str(Path(__file__).resolve())]
    raise SystemExit(stcli.main())


def ensure_streamlit() -> None:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:
        return
    if get_script_run_ctx() is None:
        if __name__ == "__main__":
            _run_streamlit_app()
        print("Run with: streamlit run damage_streamlit.py", file=sys.stderr)
        raise SystemExit(0)


ensure_streamlit()

st.set_page_config(page_title="Profiles", layout="wide")
st.markdown(
    """
    <style>
    .stDataFrame, .stDataFrame * {
        color: #000000 !important;
    }
    .stLinkButton a {
        background-color: #FF4B4B !important;
        color: #FFFFFF !important;
        border: 1px solid #FF4B4B !important;
    }
    .stLinkButton a:hover {
        background-color: #E04343 !important;
        color: #FFFFFF !important;
        border: 1px solid #E04343 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=86400, max_entries=20)
def _load_csv_cached(path_str: str, mtime: float) -> pd.DataFrame:
    path = Path(path_str)
    if not path.exists():
        return pd.DataFrame()
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
        return _optimize_dataframe_memory(df)
    df = pd.read_csv(path)
    return _optimize_dataframe_memory(df)


def _optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce DataFrame memory footprint by optimizing dtypes"""
    if df.empty:
        return df

    # Convert object columns to category where appropriate (50%+ savings)
    for col in df.select_dtypes(include=["object"]).columns:
        # Skip ID columns
        if col.endswith("_mlbid") or col.endswith("_id"):
            continue
        num_unique = df[col].nunique()
        num_total = len(df[col])
        # If less than 50% unique values, use category
        if num_unique / num_total < 0.5:
            df[col] = df[col].astype("category")

    # Downcast numeric types (30-50% savings)
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    return df


def load_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if path.suffix == ".csv":
        parquet_path = path.with_suffix(".parquet")
        if parquet_path.exists():
            path = parquet_path
    if not path.exists():
        return pd.DataFrame()
    return _load_csv_cached(str(path), path.stat().st_mtime)


@st.cache_data(ttl=86400, max_entries=3)
def load_damage_df() -> pd.DataFrame:
    # Prefer the most comprehensive file with newest data
    preferred_files = [
        DATA_DIR / "damage_pos_2015_2025.csv",
    ]
    for preferred in preferred_files:
        parquet_preferred = preferred.with_suffix(".parquet")
        if parquet_preferred.exists():
            df = pd.read_parquet(parquet_preferred)
            return _optimize_dataframe_memory(df)
        if preferred.exists():
            df = pd.read_csv(preferred)
            return _optimize_dataframe_memory(df)
    candidates = sorted(DATA_DIR.glob("damage_pos_*.parquet"))
    if candidates:
        df = pd.read_parquet(candidates[-1])
        return _optimize_dataframe_memory(df)
    candidates = sorted(DATA_DIR.glob("damage_pos_*.csv"))
    if candidates:
        df = pd.read_csv(candidates[-1])
        return _optimize_dataframe_memory(df)
    return pd.DataFrame()


def season_options(df: pd.DataFrame, column: str = "season") -> list:
    if df.empty or column not in df.columns:
        return ["All"]
    values = pd.Series(df[column].dropna().unique())
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all():
        sorted_vals = values.loc[numeric.sort_values(ascending=False).index].tolist()
    else:
        sorted_vals = values.sort_values(ascending=False).tolist()
    return ["All"] + sorted_vals


def filter_by_values(df: pd.DataFrame, column: str, values) -> pd.DataFrame:
    if df.empty:
        return df
    if values is None:
        return df
    if isinstance(values, (str, bytes)):
        if values == "All":
            return df
        return df[df[column] == values]
    if not isinstance(values, (list, tuple, set, pd.Index, np.ndarray, pd.Series)):
        return df[df[column] == values]
    values_list = list(values)
    if not values_list or "All" in values_list:
        return df
    return df[df[column].isin(values_list)]


def _split_team_tokens(value: str) -> list[str]:
    tokens = [token.strip() for token in re.split(r"[|,/]", value)]
    return [token for token in tokens if token]


def team_options(df: pd.DataFrame, column: str) -> list[str]:
    if df.empty or column not in df.columns:
        return ["All"]
    tokens: set[str] = set()
    for value in df[column].dropna().astype(str):
        tokens.update(_split_team_tokens(value))
    return ["All"] + sorted(tokens)


def filter_by_team_token(df: pd.DataFrame, column: str, team: str) -> pd.DataFrame:
    if df.empty or team == "All":
        return df
    if column not in df.columns:
        return df
    mask = df[column].astype(str).apply(lambda v: team in _split_team_tokens(v))
    return df[mask]


def position_options(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["All"]
    options = [
        pos
        for pos in POSITION_FILTER_COLS
        if pos in df.columns or f"is_{pos}" in df.columns
    ]
    return ["All"] + options if options else ["All"]


def filter_by_positions(
    df: pd.DataFrame,
    positions,
    min_count: int = POSITION_COUNT_THRESHOLD,
) -> pd.DataFrame:
    if df.empty or positions is None:
        return df
    if isinstance(positions, (str, bytes)):
        positions = [positions]
    if not isinstance(positions, (list, tuple, set, pd.Index, np.ndarray, pd.Series)):
        positions = [positions]
    selected = [pos for pos in positions if pos != "All"]
    if not selected:
        return df
    mask = pd.Series(False, index=df.index)
    for pos in selected:
        binary_col = f"is_{pos}"
        if binary_col in df.columns:
            values = pd.to_numeric(df[binary_col], errors="coerce").fillna(0)
            mask |= values >= 1
            continue
        if pos not in df.columns:
            continue
        values = pd.to_numeric(df[pos], errors="coerce").fillna(0)
        threshold = 1 if values.max(skipna=True) <= 1 else min_count
        mask |= values >= threshold
    if not mask.any():
        return df.iloc[0:0]
    return df[mask]


def player_id_options(
    df: pd.DataFrame, id_col: str, name_col: str
) -> tuple[list, dict]:
    if df.empty or id_col not in df.columns:
        return ["All"], {}
    options_df = (
        df[[id_col, name_col]].copy() if name_col in df.columns else df[[id_col]].copy()
    )
    options_df[id_col] = pd.to_numeric(options_df[id_col], errors="coerce")
    options_df = options_df.dropna(subset=[id_col])
    if name_col in options_df.columns:
        options_df[name_col] = options_df[name_col].astype(str)
    options_df = options_df.drop_duplicates(subset=[id_col])
    if name_col in options_df.columns:
        options_df = options_df.sort_values(by=[name_col, id_col])
        name_map = dict(zip(options_df[id_col], options_df[name_col]))
    else:
        options_df = options_df.sort_values(by=[id_col])
        name_map = {}
    ids = options_df[id_col].tolist()
    return ["All"] + ids, name_map


def numeric_filter(df: pd.DataFrame, column: str, min_value: float) -> pd.DataFrame:
    if df.empty:
        return df
    return df[df[column] >= min_value]


def pitcher_workload_filter(
    df: pd.DataFrame, filter_type: str, min_value: float
) -> pd.DataFrame:
    if df.empty:
        return df
    metric = filter_type if filter_type in {"IP", "TBF", "GS"} else "TBF"
    if metric not in df.columns:
        return df
    return numeric_filter(df, metric, min_value)


def download_button(df: pd.DataFrame, label: str, key: str) -> None:
    if df.empty:
        return
    if not _is_user_subscribed():
        st.info("Subscribe to download the full dataset.")
        return
    csv = df.to_csv(index=False)
    st.download_button(label, data=csv, file_name=f"{label}.csv", key=key)


def apply_column_filters(df: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    if df.empty:
        return df
    with st.expander("Column filters", expanded=False):
        filtered = df
        for col in df.columns:
            if col.startswith("__"):
                continue
            col_key = f"{key_prefix}_{col}"
            if pd.api.types.is_numeric_dtype(df[col]):
                op = st.selectbox(
                    f"{col} filter",
                    options=["(no filter)", "=", "<", "<=", ">", ">=", "between"],
                    key=f"{col_key}_op",
                )
                if op == "(no filter)":
                    continue
                if op == "between":
                    low = st.number_input(f"{col} min", key=f"{col_key}_min", value=0.0)
                    high = st.number_input(
                        f"{col} max", key=f"{col_key}_max", value=0.0
                    )
                    filtered = filtered[
                        (filtered[col] >= low) & (filtered[col] <= high)
                    ]
                else:
                    value = st.number_input(
                        f"{col} value", key=f"{col_key}_val", value=0.0
                    )
                    if op == "=":
                        filtered = filtered[filtered[col] == value]
                    elif op == "<":
                        filtered = filtered[filtered[col] < value]
                    elif op == "<=":
                        filtered = filtered[filtered[col] <= value]
                    elif op == ">":
                        filtered = filtered[filtered[col] > value]
                    elif op == ">=":
                        filtered = filtered[filtered[col] >= value]
            else:
                op = st.selectbox(
                    f"{col} filter",
                    options=["(no filter)", "=", "contains"],
                    key=f"{col_key}_op",
                )
                if op == "(no filter)":
                    continue
                value = st.text_input(f"{col} value", key=f"{col_key}_val", value="")
                if value:
                    if op == "=":
                        filtered = filtered[filtered[col] == value]
                    else:
                        filtered = filtered[
                            filtered[col]
                            .astype(str)
                            .str.contains(value, case=False, na=False)
                        ]
        return filtered


def _resolve_subscription_status(result: object | None = None) -> bool:
    if isinstance(result, bool):
        return result
    if isinstance(result, dict):
        for key in ("subscribed", "is_subscribed", "subscription_active", "active"):
            if key in result:
                return bool(result.get(key))
    for key in (
        "user_subscribed",
        "is_subscribed",
        "subscription_active",
        "subscribed",
    ):
        if key in st.session_state:
            return bool(st.session_state.get(key))
    return False


def _is_user_subscribed() -> bool:
    return _resolve_subscription_status()


def _pick_first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns and df[col].notna().any():
            return col
    return None


def _coerce_numeric_for_plot(
    df: pd.DataFrame,
    exclude_cols: set[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    plot_df = df.copy()
    numeric_cols = plot_df.select_dtypes(include="number").columns.tolist()
    if plot_df.empty:
        return plot_df, numeric_cols

    min_required = max(3, int(len(plot_df) * 0.2))
    exclude_cols = {col.lower() for col in (exclude_cols or set())}
    object_cols = [
        col
        for col in plot_df.columns
        if col not in numeric_cols
        and pd.api.types.is_object_dtype(plot_df[col])
        and col.lower() not in exclude_cols
    ]
    for col in object_cols:
        cleaned = (
            plot_df[col]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
        )
        coerced = pd.to_numeric(cleaned, errors="coerce")
        if coerced.notna().sum() >= min_required:
            plot_df[col] = coerced
            numeric_cols.append(col)

    if len(numeric_cols) < 2:
        for col in object_cols:
            if col in numeric_cols:
                continue
            cleaned = (
                plot_df[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.replace(",", "", regex=False)
            )
            coerced = pd.to_numeric(cleaned, errors="coerce")
            if coerced.notna().sum() >= 1:
                plot_df[col] = coerced
                numeric_cols.append(col)
                if len(numeric_cols) >= 2:
                    break

    return plot_df, numeric_cols


def _build_point_labels(
    df: pd.DataFrame,
    include_team: bool,
    label_cols: list[str] | None = None,
) -> pd.Series | None:
    if df.empty:
        return None
    player_col = _pick_first_col(
        df,
        [
            "Player",
            "player",
            "player_name",
            "Name",
            "Batter",
            "Pitcher",
            "Batter Name",
            "Pitcher Name",
        ],
    )
    team_col = None
    if include_team:
        team_col = _pick_first_col(
            df,
            [
                "Team",
                "team",
                "hitting_code",
                "pitching_code",
                "Team Code",
                "Team Abbrev",
            ],
        )
    pitch_col = _pick_first_col(
        df,
        [
            "Pitch Type",
            "pitch_type",
            "pitch_type_name",
            "PitchType",
            "TaggedPitchType",
            "pitch_tag",
        ],
    )
    split_col = _pick_first_col(
        df,
        [
            "split",
            "split_type",
            "Split",
            "Split Type",
        ],
    )
    if not any([player_col, team_col, pitch_col, split_col]):
        return None

    resolved_label_cols: list[str] = []
    if label_cols:
        lower_map = {col.lower(): col for col in df.columns}
        for name in label_cols:
            if name in df.columns:
                resolved_label_cols.append(name)
                continue
            key = name.lower()
            if key in lower_map:
                resolved_label_cols.append(lower_map[key])
    if include_team and team_col and team_col not in resolved_label_cols:
        resolved_label_cols.append(team_col)

    def build_label(row: pd.Series) -> str:
        if resolved_label_cols:
            parts: list[str] = []
            for col in resolved_label_cols:
                if not col:
                    continue
                value = row.get(col)
                if pd.isna(value):
                    continue
                value_str = str(value).strip()
                if value_str:
                    parts.append(value_str)
            return " | ".join(parts)

        if player_col and not include_team:
            parts: list[str] = []
            for col in [player_col, pitch_col, split_col]:
                if not col:
                    continue
                value = row.get(col)
                if pd.isna(value):
                    continue
                value_str = str(value).strip()
                if value_str:
                    parts.append(value_str)
            return " | ".join(parts)

        parts: list[str] = []
        for col in [player_col, team_col, pitch_col, split_col]:
            if not col:
                continue
            value = row.get(col)
            if pd.isna(value):
                continue
            value_str = str(value).strip()
            if value_str:
                parts.append(value_str)
        return " | ".join(parts)

    labels = df.apply(build_label, axis=1)
    if labels.str.strip().eq("").all():
        return None
    return labels


def _render_plot_controls(
    df: pd.DataFrame,
    table_key: str,
    include_team_label: bool,
    reverse_cols: set[str],
    label_cols: list[str] | None,
) -> None:
    exclude_cols = set(label_cols or [])
    plot_df, numeric_cols = _coerce_numeric_for_plot(df, exclude_cols=exclude_cols)
    if len(numeric_cols) < 2:
        st.info("Not enough numeric columns to plot.")
        return

    with st.expander("Create-a-Plot", expanded=False):
        col1, col2 = st.columns(2)
        x_col = col1.selectbox(
            "X column",
            options=numeric_cols,
            index=0,
            key=f"{table_key}_plot_x",
        )
        y_options = [col for col in numeric_cols if col != x_col]
        if not y_options:
            y_options = numeric_cols
        y_default = 1 if len(y_options) > 1 and y_options[0] == x_col else 0
        y_col = col2.selectbox(
            "Y column",
            options=y_options,
            index=y_default,
            key=f"{table_key}_plot_y",
        )
        col3, col4 = st.columns(2)
        size_options = ["(none)"] + numeric_cols
        size_col = col3.selectbox(
            "Size",
            options=size_options,
            index=0,
            key=f"{table_key}_plot_size",
        )
        color_options = ["(none)"] + numeric_cols
        color_col = col4.selectbox(
            "Color",
            options=color_options,
            index=0,
            key=f"{table_key}_plot_color",
        )
        max_points = st.number_input(
            "Max labeled points (sampled if exceeded)",
            min_value=100,
            max_value=20000,
            value=100,
            step=100,
            key=f"{table_key}_plot_max",
        )
        show_labels = st.checkbox(
            "Show point labels",
            value=True,
            key=f"{table_key}_plot_labels",
        )
        st.caption(
            "Large point counts will disable labels. Lower Max points to show labels."
        )

        plot_df = plot_df.copy()

        size_arg = None if size_col == "(none)" else size_col
        color_arg = None if color_col == "(none)" else color_col
        colorscale = None
        color_midpoint = None
        if color_arg is not None:
            colorscale = "RdYlGn_r" if color_arg in reverse_cols else "RdYlGn"
            color_midpoint = float(plot_df[color_arg].median())

        plot_df = plot_df.reset_index(drop=True)
        if x_col in plot_df.columns and y_col in plot_df.columns:
            plot_df = plot_df.copy()
            plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors="coerce")
            plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
            plot_df = plot_df.dropna(subset=[x_col, y_col])
            if plot_df.empty:
                st.info("No rows available after numeric coercion of X/Y.")
                return
            q10_x = plot_df[x_col].quantile(0.1)
            q90_x = plot_df[x_col].quantile(0.9)
            q10_y = plot_df[y_col].quantile(0.1)
            q90_y = plot_df[y_col].quantile(0.9)
            extremes = (
                (plot_df[x_col] <= q10_x)
                | (plot_df[x_col] >= q90_x)
                | (plot_df[y_col] <= q10_y)
                | (plot_df[y_col] >= q90_y)
            )
            if show_labels:
                labels = _build_point_labels(
                    plot_df,
                    include_team=include_team_label,
                    label_cols=label_cols,
                )
                if labels is not None:
                    label_mask = extremes.copy()
                    if label_mask.sum() > max_points:
                        sampled_idx = (
                            plot_df[label_mask]
                            .sample(n=int(max_points), random_state=0)
                            .index
                        )
                        label_mask = plot_df.index.isin(sampled_idx)
                    plot_df = plot_df.copy()
                    plot_df["__label"] = labels.where(label_mask, "")

        fig = px.scatter(
            plot_df,
            x=x_col,
            y=y_col,
            size=size_arg,
            color=color_arg,
            text="__label" if show_labels and "__label" in plot_df.columns else None,
            hover_name="__label" if "__label" in plot_df.columns else None,
            color_continuous_scale=colorscale,
            color_continuous_midpoint=color_midpoint,
            render_mode="svg",
        )
        if show_labels and "__label" in plot_df.columns:
            fig.update_traces(textposition="top center", mode="markers+text")
        else:
            fig.update_traces(mode="markers")
        fig.update_traces(marker=dict(size=7, opacity=0.7))
        fig.update_layout(
            margin=dict(l=10, r=10, t=30, b=10),
            height=520,
        )
        st.plotly_chart(fig, width="stretch")


def render_table(
    df: pd.DataFrame,
    reverse_cols: set[str] | None = None,
    no_format_cols: set[str] | None = None,
    group_cols: list[str] | None = None,
    stats_df: pd.DataFrame | None = None,
    abs_cols: set[str] | None = None,
    show_controls: bool = True,
    include_team_label: bool = False,
    label_cols: list[str] | None = None,
    hide_cols: set[str] | None = None,
    round_decimals: int = 1,
) -> None:
    if df.empty:
        st.info("No data available yet.")
        return
    if not _is_user_subscribed():
        st.info(
            f"Preview mode: showing the first {PREVIEW_ROWS} rows. Subscribe for full access."
        )
        df = df.head(PREVIEW_ROWS)
        show_controls = False

    global _TABLE_COUNTER
    table_key = f"table_{_TABLE_COUNTER}"
    _TABLE_COUNTER += 1

    if show_controls:
        df = apply_column_filters(df, table_key)
        if df.empty:
            st.info("No data after filters.")
            return

    def _contains_non_mlb_rows(table_df: pd.DataFrame) -> bool:
        level_cols = ["__level", "level_id", "Level"]
        for col in level_cols:
            if col not in table_df.columns:
                continue
            s = table_df[col]
            if col in {"__level", "level_id"}:
                vals = pd.to_numeric(s, errors="coerce")
                if vals.notna().any() and (vals != 1).any():
                    return True
            else:
                text = s.astype(str).str.strip().str.lower()
                milb_tokens = {"triple-a", "low-a", "low minors"}
                if text.isin(milb_tokens).any():
                    return True
        return False

    hide_cols = hide_cols or set()
    # Hide Player ID by default from display while keeping it for downloads.
    hide_cols = set(hide_cols) | {"Player ID"}
    # Hide Team column whenever the displayed table contains any non-MLB rows.
    if _contains_non_mlb_rows(df):
        hide_cols = set(hide_cols) | {"Team"}
    display_cols = [
        col for col in df.columns if not col.startswith("__") and col not in hide_cols
    ]
    df_display = df[display_cols].copy()

    _render_plot_controls(
        df_display,
        table_key,
        include_team_label,
        reverse_cols or set(),
        label_cols,
    )

    if show_controls:
        page_size_option = st.selectbox(
            "Rows per page",
            options=["All", 25, 50, 100, 200],
            index=2,
            key=f"{table_key}_page_size",
        )
        total_rows = len(df_display)
        if page_size_option == "All":
            page_size = total_rows
            page = 1
        else:
            page_size = int(page_size_option)
            total_pages = max(1, (total_rows + page_size - 1) // page_size)
            page = st.number_input(
                "Page",
                min_value=1,
                max_value=int(total_pages),
                value=1,
                step=1,
                key=f"{table_key}_page",
            )

        start = (page - 1) * page_size
        end = start + page_size
        df_page_display = df_display.iloc[start:end].copy()
        df_page_full = df.iloc[start:end].copy()
    else:
        df_page_display = df_display.copy()
        df_page_full = df.copy()

    max_elements = pd.get_option("styler.render.max_elements")
    total_cells = df_page_display.shape[0] * df_page_display.shape[1]
    reverse_cols = reverse_cols or set()
    abs_cols = abs_cols or set()
    no_format_cols = no_format_cols or DEFAULT_NO_FORMAT_COLS
    numeric_cols = df_display.select_dtypes(include="number").columns
    float_cols = df.select_dtypes(include="floating").columns
    format_cols = [col for col in numeric_cols if col not in no_format_cols]

    if len(numeric_cols) > 0:
        df_page_display[numeric_cols] = df_page_display[numeric_cols].round(
            round_decimals
        )
    if len(float_cols) > 0:
        df_page_display[float_cols] = df_page_display[float_cols].round(round_decimals)

    if len(format_cols) > 0 and total_cells <= max_elements:
        stats_source = stats_df if stats_df is not None else df
        similarity_cols = [col for col in format_cols if col.startswith("Similarity")]
        stats_format_cols = [col for col in format_cols if col in stats_source.columns]
        if not stats_format_cols and not similarity_cols:
            st.dataframe(df_page_display, width="stretch", hide_index=True)
            return
        similarity_medians: dict[str, float] = {}
        for col in similarity_cols:
            if col in stats_source.columns:
                similarity_medians[col] = stats_source[col].median()
            else:
                similarity_medians[col] = df[col].median()
        group_cols = group_cols or []
        group_cols = [col for col in group_cols if col in stats_source.columns]
        abs_format_cols = [col for col in abs_cols if col in stats_source.columns]
        abs_stats_source = stats_source.copy()
        if abs_format_cols:
            abs_stats_source[abs_format_cols] = abs_stats_source[abs_format_cols].abs()
        if group_cols:
            if stats_format_cols:
                q10 = stats_source.groupby(group_cols)[stats_format_cols].quantile(0.05)
                q90 = stats_source.groupby(group_cols)[stats_format_cols].quantile(0.95)
                med = stats_source.groupby(group_cols)[stats_format_cols].median()
            else:
                q10 = q90 = med = None
            if abs_format_cols:
                q10_abs = abs_stats_source.groupby(group_cols)[
                    abs_format_cols
                ].quantile(0.05)
                q90_abs = abs_stats_source.groupby(group_cols)[
                    abs_format_cols
                ].quantile(0.95)
                med_abs = abs_stats_source.groupby(group_cols)[abs_format_cols].median()
            else:
                q10_abs = q90_abs = med_abs = None
        else:
            if stats_format_cols:
                q10 = stats_source[stats_format_cols].quantile(0.05)
                q90 = stats_source[stats_format_cols].quantile(0.95)
                med = stats_source[stats_format_cols].median()
            else:
                q10 = q90 = med = None
            if abs_format_cols:
                q10_abs = abs_stats_source[abs_format_cols].quantile(0.05)
                q90_abs = abs_stats_source[abs_format_cols].quantile(0.95)
                med_abs = abs_stats_source[abs_format_cols].median()
            else:
                q10_abs = q90_abs = med_abs = None
        cmap = colors.LinearSegmentedColormap.from_list(
            "rwgn", ["#c75c5c", "#f7f7f7", "#5cb85c"]
        )
        cmap_rev = colors.LinearSegmentedColormap.from_list(
            "gnrw", ["#5cb85c", "#f7f7f7", "#c75c5c"]
        )
        alpha = 0.9

        def style_row(row: pd.Series) -> list[str]:
            if group_cols:
                if q10 is None:
                    row_q10 = row_q90 = row_med = None
                    row_q10_abs = row_q90_abs = row_med_abs = None
                else:
                    group_vals = df_page_full.loc[row.name, group_cols]
                    if isinstance(group_vals, pd.Series):
                        group_key = tuple(group_vals.values.tolist())
                    else:
                        group_key = group_vals
                    if group_key not in q10.index:
                        return [""] * len(row)
                    row_q10 = q10.loc[group_key]
                    row_q90 = q90.loc[group_key]
                    row_med = med.loc[group_key]
                    if q10_abs is None:
                        row_q10_abs = row_q90_abs = row_med_abs = None
                    else:
                        row_q10_abs = q10_abs.loc[group_key]
                        row_q90_abs = q90_abs.loc[group_key]
                        row_med_abs = med_abs.loc[group_key]
            else:
                row_q10 = q10
                row_q90 = q90
                row_med = med
                row_q10_abs = q10_abs
                row_q90_abs = q90_abs
                row_med_abs = med_abs

            styles: list[str] = []
            for col in row.index:
                if col not in format_cols:
                    styles.append("")
                    continue
                if col in similarity_medians:
                    vmin = 0
                    vmax = 99
                    vcenter = similarity_medians[col]
                else:
                    if col not in stats_format_cols or row_q10 is None:
                        styles.append("")
                        continue
                    if col in abs_cols and row_q10_abs is not None:
                        vmin = row_q10_abs[col]
                        vmax = row_q90_abs[col]
                        vcenter = row_med_abs[col]
                    else:
                        vmin = row_q10[col]
                        vmax = row_q90[col]
                        vcenter = row_med[col]
                if pd.isna(vmin) or pd.isna(vmax) or vmin == vmax:
                    styles.append("")
                    continue
                if pd.isna(vcenter):
                    styles.append("")
                    continue
                # TwoSlopeNorm requires strict ordering: vmin < vcenter < vmax.
                if not (vmin < vcenter < vmax):
                    vcenter = (vmin + vmax) / 2
                if not (vmin < vcenter < vmax):
                    styles.append("")
                    continue
                norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
                val = row[col]
                if pd.isna(val):
                    styles.append("")
                    continue
                if col in abs_cols:
                    val = abs(val)
                val = float(np.clip(val, vmin, vmax))
                col_cmap = cmap_rev if col in reverse_cols else cmap
                rgb = colors.to_rgb(col_cmap(norm(val)))
                styles.append(
                    "background-color: "
                    f"rgba({int(rgb[0] * 255)},{int(rgb[1] * 255)},{int(rgb[2] * 255)},{alpha}); color: #000000"
                )
            return styles

        styler = df_page_display.style.apply(style_row, axis=1)
        if len(float_cols) > 0:
            format_map = {col: f"{{:.{round_decimals}f}}" for col in float_cols}
            # Format integer-value columns without decimals
            int_keywords = ["Similarity", "Pitch Grade", "BB Spin", "Pctile", "#"]
            for col in df_page_display.columns:
                if any(kw in col for kw in int_keywords):
                    format_map[col] = "{:.0f}"
            styler = styler.format(format_map)
        st.dataframe(styler, width="stretch", hide_index=True)
        return
    if len(float_cols) > 0:
        # Identify columns that should display as integers
        int_keywords = ["Similarity", "Pitch Grade", "BB Spin", "Pctile", "#"]
        int_cols = [
            col
            for col in df_page_display.columns
            if any(kw in col for kw in int_keywords)
        ]
        other_float_cols = [col for col in float_cols if col not in int_cols]

        if other_float_cols:
            df_page_display[other_float_cols] = df_page_display[
                other_float_cols
            ].applymap(lambda x: f"{x:.{round_decimals}f}" if pd.notna(x) else x)
        for col in int_cols:
            if col in df_page_display.columns:
                df_page_display[col] = df_page_display[col].apply(
                    lambda x: f"{x:.0f}" if pd.notna(x) else x
                )
    st.dataframe(df_page_display, width="stretch", hide_index=True)


# Load datasets

damage_df = load_damage_df()
hitter_pct = load_csv("hitter_pctiles.csv")
pitcher_df = load_csv("pitcher_stuff_new.csv")
pitcher_pct = load_csv("pitcher_pctiles.csv")
hitting_avg = load_csv("new_hitting_lg_avg.csv")
pitching_avg = load_csv("new_lg_stuff.csv")
team_damage = load_csv("new_team_damage.csv")
team_stuff = load_csv("new_team_stuff.csv")
pitch_types = load_csv("new_pitch_types.csv")
pitch_types_pct = load_csv("pitch_types_pctiles.csv")
hitters_regressed = load_csv("hitters_regressed.csv")
pitchers_regressed = load_csv("pitchers_regressed.csv")
pitch_types_regressed = load_csv("pitch_types_regressed.csv")
hitter_splits_df = load_csv("hitter_splits.csv")
pitcher_splits_df = load_csv("pitcher_splits.csv")
pitch_type_splits_df = load_csv("pitch_types_splits.csv")
league_pitch_types = load_csv("league_pitch_types.csv")
park_data = load_csv("park_data.csv")


# Normalize team column names: new CSVs use "team", old use "pitching_code"/"hitting_code"
def _normalize_team_col(df: pd.DataFrame, old_col: str) -> pd.DataFrame:
    """If 'team' column exists, rename it to old_col for backward compatibility."""
    if df.empty:
        return df
    if "team" in df.columns and old_col not in df.columns:
        return df.rename(columns={"team": old_col})
    return df


def _normalize_la_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Rename old FB_pct/GB_pct columns to new LA_gte_20/LA_lte_0 names."""
    if df.empty:
        return df
    rename_map = {}
    if "FB_pct" in df.columns and "LA_gte_20" not in df.columns:
        rename_map["FB_pct"] = "LA_gte_20"
    if "GB_pct" in df.columns and "LA_lte_0" not in df.columns:
        rename_map["GB_pct"] = "LA_lte_0"
    if rename_map:
        return df.rename(columns=rename_map)
    return df


def _normalize_split_cols(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    for col in ["split_type", "split"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    return df


def _similarity_choice_labels(
    df: pd.DataFrame,
    display_map: dict[str, str],
    exclude_cols: set[str],
) -> tuple[list[str], dict[str, str]]:
    if df.empty:
        return [], {}
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    filtered = []
    for col in numeric_cols:
        if col in exclude_cols:
            continue
        if col == "reg_prop":
            continue
        if col.endswith("_raw") or col.endswith("_raw_reg"):
            continue
        if "_num" in col or "_den" in col or "_n" in col:
            continue
        if col.endswith("_id") or col.endswith("_mlbid"):
            continue
        filtered.append(col)
    labels = {}
    for col in filtered:
        label = display_map.get(col, col.replace("_", " ").title())
        if label.endswith(" Reg"):
            label = label[: -len(" Reg")]
        labels[col] = label
    return filtered, labels


def _merge_regressed(
    base_df: pd.DataFrame, reg_df: pd.DataFrame, keys: list[str]
) -> pd.DataFrame:
    if base_df.empty or reg_df.empty:
        return pd.DataFrame()
    reg_cols = [
        c
        for c in reg_df.columns
        if c.endswith("_reg") or c.endswith("_raw") or c.endswith("_n")
    ]
    keep_cols = list(dict.fromkeys(keys + reg_cols))
    reg_small = reg_df[keep_cols].drop_duplicates(subset=keys)
    # Normalize key dtypes before merge to avoid pandas factorizer crashes
    # with mixed/downcast integer dtypes (e.g. int8/int16/int32).
    left = base_df.copy()
    right = reg_small.copy()
    for key in keys:
        if key not in left.columns or key not in right.columns:
            continue
        left_key = left[key]
        right_key = right[key]
        if pd.api.types.is_numeric_dtype(left_key) or pd.api.types.is_numeric_dtype(
            right_key
        ):
            left[key] = pd.to_numeric(left_key, errors="coerce").astype("Int64")
            right[key] = pd.to_numeric(right_key, errors="coerce").astype("Int64")
        else:
            left[key] = left_key.astype("string")
            right[key] = right_key.astype("string")
    return left.merge(right, on=keys, how="left")


def _hitter_display_map(include_mlb_eq: bool = False) -> dict[str, str]:
    display_map = {
        "hitter_name": "Name",
        "hitting_code": "Team",
        "season": "Season",
        "bbe": "BBE",
        "damage_rate_reg": "Damage/BBE (%)",
        "EV90th_reg": "90th Pctile EV",
        "pull_FB_pct_reg": "Pulled FB (%)",
        "selection_skill_reg": "Selectivity (%)",
        "hittable_pitches_taken_reg": "Hittable Pitch Take (%)",
        "chase_reg": "Chase (%)",
        "z_con_reg": "Z-Contact (%)",
        "secondary_whiff_pct_reg": "Whiff vs. Secondaries (%)",
        "similarity_score": "Similarity (0-100)",
        "LA_gte_20_reg": "LA>=20 (%)",
        "LA_lte_0_reg": "LA<=0%",
        "SEAGER_reg": "SEAGER",
        "whiffs_vs_95_reg": "Whiff vs. 95+ (%)",
        "contact_vs_avg_reg": "Contact Over Expected (%)",
        "LD_pct_reg": "0<LA<20 (%)",
        "bat_speed_reg": "Bat Speed",
        "swing_length_reg": "Swing Length",
        "attack_angle_reg": "Attack Angle",
        "swing_path_tilt_reg": "VBA",
        "max_EV_reg": "Max EV",
    }
    if not include_mlb_eq:
        return display_map
    eq_map = {}
    for col, label in display_map.items():
        if col.endswith("_reg"):
            eq_map[f"{col}_mlb_eq"] = f"{label} MLB Eq"
    return {**display_map, **eq_map}


def _pitcher_display_map(include_mlb_eq: bool = False) -> dict[str, str]:
    display_map = {
        "name": "Name",
        "pitching_code": "Team",
        "season": "Season",
        "stuff": "Pitch Grade",
        "stuff_raw_reg": "Pitch Grade (Raw Model)",
        "fastball_velo_reg": "FA mph",
        "max_velo_reg": "Max FA mph",
        "fastball_vaa_reg": "FA VAA",
        "loc_adj_vaa_reg": "Loc-Adj VAA",
        "SwStr_reg": "SwStr (%)",
        "Ball_pct_reg": "Ball (%)",
        "Chase_reg": "Chase (%)",
        "Z_Contact_reg": "Z-Contact (%)",
        "Zone_reg": "Zone (%)",
        "CSW_reg": "CSW (%)",
        "pWhiff_reg": "pWhiff (%)",
        "FA_pct_reg": "FA (%)",
        "BB_rpm_reg": "BB RPM",
        "FA_spin_eff_reg": "FA Spin Efficiency (%)",
        "LA_lte_0_reg": "LA<=0%",
        "LD_pct_reg": "0<LA<20 (%)",
        "LA_gte_20_reg": "LA>=20 (%)",
        "rel_z_reg": "Vertical Release (ft.)",
        "rel_x_reg": "Horizontal Release (ft.)",
        "ext_reg": "Extension (ft.)",
        "arm_angle_reg": "Arm Angle",
        "similarity_score": "Similarity (0-100)",
    }
    if not include_mlb_eq:
        return display_map
    eq_map: dict[str, str] = {}
    for col, label in display_map.items():
        if col.endswith("_reg") or col == "stuff":
            eq_map[f"{col}_mlb_eq"] = f"{label} MLB Eq"
    return {**display_map, **eq_map}


def _compose_linear(a1: float, b1: float, a2: float, b2: float) -> tuple[float, float]:
    return a2 + (b2 * a1), b2 * b1


def _weighted_linear_fit(
    x: np.ndarray, y: np.ndarray, w: np.ndarray
) -> tuple[float, float, int]:
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    n = int(valid.sum())
    if n < 2:
        return np.nan, np.nan, n
    xv = x[valid]
    yv = y[valid]
    wv = w[valid]
    design = np.column_stack([np.ones(len(xv)), xv])
    sqrt_w = np.sqrt(wv)
    try:
        coef, *_ = np.linalg.lstsq(design * sqrt_w[:, None], yv * sqrt_w, rcond=None)
    except Exception:
        return np.nan, np.nan, n
    return float(coef[0]), float(coef[1]), n


def _build_hitter_mlb_equivalencies(
    base_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if base_df.empty:
        return pd.DataFrame(), pd.DataFrame(), []
    if "PA" not in base_df.columns:
        return base_df.copy(), pd.DataFrame(), []

    metric_cols = [
        col
        for col in base_df.select_dtypes(include="number").columns
        if col.endswith("_reg") and col != "reg_prop"
    ]
    if not metric_cols:
        return base_df.copy(), pd.DataFrame(), []

    keys = ["batter_mlbid", "season", "level_id"]
    base_cols = [col for col in keys + ["PA"] + metric_cols if col in base_df.columns]
    fit_df = base_df[base_cols].copy()
    grouped = fit_df.groupby(keys, as_index=False).agg(
        {"PA": "sum", **{col: "mean" for col in metric_cols}}
    )

    means = grouped.groupby(["season", "level_id"])[metric_cols].mean()
    stds = grouped.groupby(["season", "level_id"])[metric_cols].std(ddof=0)
    means = means.add_suffix("__mean")
    stds = stds.add_suffix("__std")
    moments = means.join(stds).reset_index()

    z_df = grouped.merge(moments, on=["season", "level_id"], how="left")
    for col in metric_cols:
        std_col = f"{col}__std"
        z_df[f"{col}__z"] = (z_df[col] - z_df[f"{col}__mean"]) / z_df[std_col].replace(
            0, np.nan
        )

    src = z_df.rename(
        columns={
            "season": "src_season",
            "level_id": "src_level",
            "PA": "src_PA",
            **{f"{col}__z": f"src_{col}__z" for col in metric_cols},
        }
    )
    dst = z_df.rename(
        columns={
            "season": "dst_season",
            "level_id": "dst_level",
            "PA": "dst_PA",
            **{f"{col}__z": f"dst_{col}__z" for col in metric_cols},
        }
    )
    # Train on both same-season transitions and adjacent-season transitions
    # (season n -> season n+1) for each level edge.
    same_pairs = src.merge(
        dst,
        left_on=["batter_mlbid", "src_season"],
        right_on=["batter_mlbid", "dst_season"],
        how="inner",
    ).assign(pair_group="same_season")
    dst_next = dst.copy()
    dst_next["src_season"] = dst_next["dst_season"] - 1
    next_pairs = src.merge(
        dst_next,
        on=["batter_mlbid", "src_season"],
        how="inner",
    ).assign(pair_group="next_season")
    pairs = pd.concat([same_pairs, next_pairs], ignore_index=True, sort=False)
    pairs = pairs[pairs["src_level"] > pairs["dst_level"]]

    edge_thresholds: dict[tuple[int, int], tuple[int, int]] = {
        (11, 1): (50, 50),
        (14, 11): (50, 50),
        (16, 14): (10, 10),
    }
    prior_a = 0.0
    prior_b = 0.5
    shrink_k = 50.0
    edge_coeff: dict[tuple[int, int, str], tuple[float, float, int]] = {}
    coeff_rows: list[dict[str, object]] = []

    for (src_level, dst_level), (min_src_pa, min_dst_pa) in edge_thresholds.items():
        edge_df = pairs[
            (pairs["src_level"] == src_level)
            & (pairs["dst_level"] == dst_level)
            & (pairs["src_PA"] >= min_src_pa)
            & (pairs["dst_PA"] >= min_dst_pa)
        ].copy()
        if edge_df.empty:
            for col in metric_cols:
                edge_coeff[(src_level, dst_level, col)] = (prior_a, prior_b, 0)
                coeff_rows.append(
                    {
                        "src_level": src_level,
                        "dst_level": dst_level,
                        "metric": col,
                        "a": prior_a,
                        "b": prior_b,
                        "n": 0,
                        "min_src_pa": min_src_pa,
                        "min_dst_pa": min_dst_pa,
                        "fit_type": "intra+inter-season",
                    }
                )
            continue

        weights = np.sqrt(
            np.clip(edge_df["src_PA"].to_numpy(dtype=float), 0, None)
            * np.clip(edge_df["dst_PA"].to_numpy(dtype=float), 0, None)
        )
        for col in metric_cols:
            x = edge_df[f"src_{col}__z"].to_numpy(dtype=float)
            y = edge_df[f"dst_{col}__z"].to_numpy(dtype=float)
            raw_a, raw_b, n = _weighted_linear_fit(x, y, weights)
            if not np.isfinite(raw_a):
                raw_a = prior_a
            if not np.isfinite(raw_b):
                raw_b = prior_b
            reliability = n / (n + shrink_k) if n > 0 else 0.0
            fit_a = reliability * raw_a + (1.0 - reliability) * prior_a
            fit_b = reliability * raw_b + (1.0 - reliability) * prior_b
            fit_a = float(np.clip(fit_a, -1.5, 1.5))
            fit_b = float(np.clip(fit_b, -0.25, 1.25))
            edge_coeff[(src_level, dst_level, col)] = (fit_a, fit_b, n)
            coeff_rows.append(
                {
                    "src_level": src_level,
                    "dst_level": dst_level,
                    "metric": col,
                    "a": fit_a,
                    "b": fit_b,
                    "n": n,
                    "min_src_pa": min_src_pa,
                    "min_dst_pa": min_dst_pa,
                    "fit_type": "intra+inter-season",
                }
            )

    level_mlb_coeff: dict[tuple[int, str], tuple[float, float, int]] = {}
    for col in metric_cols:
        a11, b11, n11 = edge_coeff[(11, 1, col)]
        a14, b14, n14 = edge_coeff[(14, 11, col)]
        a16, b16, n16 = edge_coeff[(16, 14, col)]
        a14_to_1, b14_to_1 = _compose_linear(a14, b14, a11, b11)
        a16_to_11, b16_to_11 = _compose_linear(a16, b16, a14, b14)
        a16_to_1, b16_to_1 = _compose_linear(a16_to_11, b16_to_11, a11, b11)
        level_mlb_coeff[(1, col)] = (0.0, 1.0, n11)
        level_mlb_coeff[(11, col)] = (a11, b11, n11)
        level_mlb_coeff[(14, col)] = (a14_to_1, b14_to_1, min(n14, n11))
        level_mlb_coeff[(16, col)] = (a16_to_1, b16_to_1, min(n16, n14, n11))
        coeff_rows.extend(
            [
                {
                    "src_level": 14,
                    "dst_level": 1,
                    "metric": col,
                    "a": a14_to_1,
                    "b": b14_to_1,
                    "n": min(n14, n11),
                    "min_src_pa": 50,
                    "min_dst_pa": 50,
                    "fit_type": "chained",
                },
                {
                    "src_level": 16,
                    "dst_level": 1,
                    "metric": col,
                    "a": a16_to_1,
                    "b": b16_to_1,
                    "n": min(n16, n14, n11),
                    "min_src_pa": 10,
                    "min_dst_pa": 50,
                    "fit_type": "chained",
                },
            ]
        )

    mlb_moments = moments[moments["level_id"] == 1].drop(columns=["level_id"]).copy()
    mlb_moments = mlb_moments.rename(
        columns={
            f"{col}__mean": f"{col}__mlb_mean"
            for col in metric_cols
            if f"{col}__mean" in mlb_moments.columns
        }
        | {
            f"{col}__std": f"{col}__mlb_std"
            for col in metric_cols
            if f"{col}__std" in mlb_moments.columns
        }
    )

    out = base_df.copy()
    out = out.merge(moments, on=["season", "level_id"], how="left")
    out = out.merge(mlb_moments, on=["season"], how="left")

    for col in metric_cols:
        src_mean_col = f"{col}__mean"
        src_std_col = f"{col}__std"
        mlb_mean_col = f"{col}__mlb_mean"
        mlb_std_col = f"{col}__mlb_std"
        if (
            col not in out.columns
            or src_mean_col not in out.columns
            or src_std_col not in out.columns
            or mlb_mean_col not in out.columns
            or mlb_std_col not in out.columns
        ):
            continue
        src_z = (out[col] - out[src_mean_col]) / out[src_std_col].replace(0, np.nan)
        a_map = {
            level_id: level_mlb_coeff.get((level_id, col), (prior_a, prior_b, 0))[0]
            for level_id in LEVEL_LABELS
        }
        b_map = {
            level_id: level_mlb_coeff.get((level_id, col), (prior_a, prior_b, 0))[1]
            for level_id in LEVEL_LABELS
        }
        pred_z = out["level_id"].map(a_map) + (out["level_id"].map(b_map) * src_z)
        pred = out[mlb_mean_col] + (pred_z * out[mlb_std_col])
        mlb_mask = out["level_id"] == 1
        non_mlb_mask = ~mlb_mask
        if col in HITTER_HIGHER_IS_WORSE_METRICS:
            pred.loc[non_mlb_mask] = np.maximum(
                pred.loc[non_mlb_mask],
                out.loc[non_mlb_mask, col],
            )
        else:
            pred.loc[non_mlb_mask] = np.minimum(
                pred.loc[non_mlb_mask],
                out.loc[non_mlb_mask, col],
            )
        # Enforce a minimum directional shift (relative to source) for key MLB
        # equivalency metrics, anchored to season-level source-vs-MLB mean gaps.
        direction = HITTER_MLB_DIRECTION_MAP.get(col)
        if direction in {"up", "down"}:
            shift_scale = HITTER_MLB_MIN_SHIFT_SCALE_OVERRIDES.get(
                col, HITTER_MLB_MIN_SHIFT_SCALE
            )
            shift = (
                (out[src_mean_col] - out[mlb_mean_col]).abs() * shift_scale
            ).fillna(0.0)
            shift_floor = HITTER_MLB_MIN_SHIFT_FLOOR.get(col, 0.0)
            if shift_floor > 0:
                shift = shift.clip(lower=shift_floor)
            if direction == "up":
                pred.loc[non_mlb_mask] = np.maximum(
                    pred.loc[non_mlb_mask],
                    out.loc[non_mlb_mask, col] + shift.loc[non_mlb_mask],
                )
            else:
                pred.loc[non_mlb_mask] = np.minimum(
                    pred.loc[non_mlb_mask],
                    out.loc[non_mlb_mask, col] - shift.loc[non_mlb_mask],
                )
        pred.loc[mlb_mask] = out.loc[mlb_mask, col]
        out[f"{col}_mlb_eq"] = pred

    helper_cols: list[str] = []
    for col in metric_cols:
        helper_cols.extend(
            [f"{col}__mean", f"{col}__std", f"{col}__mlb_mean", f"{col}__mlb_std"]
        )
    out = out.drop(columns=[col for col in helper_cols if col in out.columns])

    coeff_df = pd.DataFrame(coeff_rows)
    return out, coeff_df, metric_cols


def _build_pitcher_mlb_equivalencies(
    base_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if base_df.empty:
        return pd.DataFrame(), pd.DataFrame(), []
    if "TBF" not in base_df.columns:
        return base_df.copy(), pd.DataFrame(), []

    metric_cols = [
        col
        for col in base_df.select_dtypes(include="number").columns
        if col.endswith("_reg")
        and col != "reg_prop"
        and col not in PITCHER_MLB_PASS_THROUGH_COLS
    ]
    if not metric_cols:
        out = base_df.copy()
        for col in PITCHER_MLB_PASS_THROUGH_COLS:
            if col in out.columns:
                out[f"{col}_mlb_eq"] = out[col]
        pass_through_metrics = [
            col
            for col in PITCHER_MLB_PASS_THROUGH_COLS
            if f"{col}_mlb_eq" in out.columns
        ]
        return out, pd.DataFrame(), pass_through_metrics

    keys = ["pitcher_mlbid", "season", "level_id"]
    base_cols = [col for col in keys + ["TBF"] + metric_cols if col in base_df.columns]
    fit_df = base_df[base_cols].copy()
    grouped = fit_df.groupby(keys, as_index=False).agg(
        {"TBF": "sum", **{col: "mean" for col in metric_cols}}
    )

    means = grouped.groupby(["season", "level_id"])[metric_cols].mean()
    stds = grouped.groupby(["season", "level_id"])[metric_cols].std(ddof=0)
    means = means.add_suffix("__mean")
    stds = stds.add_suffix("__std")
    moments = means.join(stds).reset_index()

    z_df = grouped.merge(moments, on=["season", "level_id"], how="left")
    for col in metric_cols:
        std_col = f"{col}__std"
        z_df[f"{col}__z"] = (z_df[col] - z_df[f"{col}__mean"]) / z_df[std_col].replace(
            0, np.nan
        )

    src = z_df.rename(
        columns={
            "season": "src_season",
            "level_id": "src_level",
            "TBF": "src_TBF",
            **{f"{col}__z": f"src_{col}__z" for col in metric_cols},
        }
    )
    dst = z_df.rename(
        columns={
            "season": "dst_season",
            "level_id": "dst_level",
            "TBF": "dst_TBF",
            **{f"{col}__z": f"dst_{col}__z" for col in metric_cols},
        }
    )
    same_pairs = src.merge(
        dst,
        left_on=["pitcher_mlbid", "src_season"],
        right_on=["pitcher_mlbid", "dst_season"],
        how="inner",
    ).assign(pair_group="same_season")
    dst_next = dst.copy()
    dst_next["src_season"] = dst_next["dst_season"] - 1
    next_pairs = src.merge(
        dst_next,
        on=["pitcher_mlbid", "src_season"],
        how="inner",
    ).assign(pair_group="next_season")
    pairs = pd.concat([same_pairs, next_pairs], ignore_index=True, sort=False)
    pairs = pairs[pairs["src_level"] > pairs["dst_level"]]

    edge_thresholds: dict[tuple[int, int], tuple[int, int]] = {
        (11, 1): (60, 60),
        (14, 11): (60, 60),
        (16, 14): (60, 60),
    }
    prior_a = 0.0
    prior_b = 0.5
    shrink_k = 50.0
    edge_coeff: dict[tuple[int, int, str], tuple[float, float, int]] = {}
    coeff_rows: list[dict[str, object]] = []

    for (src_level, dst_level), (min_src_tbf, min_dst_tbf) in edge_thresholds.items():
        edge_df = pairs[
            (pairs["src_level"] == src_level)
            & (pairs["dst_level"] == dst_level)
            & (pairs["src_TBF"] >= min_src_tbf)
            & (pairs["dst_TBF"] >= min_dst_tbf)
        ].copy()
        if edge_df.empty:
            for col in metric_cols:
                edge_coeff[(src_level, dst_level, col)] = (prior_a, prior_b, 0)
                coeff_rows.append(
                    {
                        "src_level": src_level,
                        "dst_level": dst_level,
                        "metric": col,
                        "a": prior_a,
                        "b": prior_b,
                        "n": 0,
                        "min_src_tbf": min_src_tbf,
                        "min_dst_tbf": min_dst_tbf,
                        "fit_type": "intra+inter-season",
                    }
                )
            continue

        weights = np.sqrt(
            np.clip(edge_df["src_TBF"].to_numpy(dtype=float), 0, None)
            * np.clip(edge_df["dst_TBF"].to_numpy(dtype=float), 0, None)
        )
        for col in metric_cols:
            x = edge_df[f"src_{col}__z"].to_numpy(dtype=float)
            y = edge_df[f"dst_{col}__z"].to_numpy(dtype=float)
            raw_a, raw_b, n = _weighted_linear_fit(x, y, weights)
            if not np.isfinite(raw_a):
                raw_a = prior_a
            if not np.isfinite(raw_b):
                raw_b = prior_b
            reliability = n / (n + shrink_k) if n > 0 else 0.0
            fit_a = reliability * raw_a + (1.0 - reliability) * prior_a
            fit_b = reliability * raw_b + (1.0 - reliability) * prior_b
            fit_a = float(np.clip(fit_a, -1.5, 1.5))
            fit_b = float(np.clip(fit_b, -0.25, 1.25))
            edge_coeff[(src_level, dst_level, col)] = (fit_a, fit_b, n)
            coeff_rows.append(
                {
                    "src_level": src_level,
                    "dst_level": dst_level,
                    "metric": col,
                    "a": fit_a,
                    "b": fit_b,
                    "n": n,
                    "min_src_tbf": min_src_tbf,
                    "min_dst_tbf": min_dst_tbf,
                    "fit_type": "intra+inter-season",
                }
            )

    level_mlb_coeff: dict[tuple[int, str], tuple[float, float, int]] = {}
    for col in metric_cols:
        a11, b11, n11 = edge_coeff[(11, 1, col)]
        a14, b14, n14 = edge_coeff[(14, 11, col)]
        a16, b16, n16 = edge_coeff[(16, 14, col)]
        a14_to_1, b14_to_1 = _compose_linear(a14, b14, a11, b11)
        a16_to_11, b16_to_11 = _compose_linear(a16, b16, a14, b14)
        a16_to_1, b16_to_1 = _compose_linear(a16_to_11, b16_to_11, a11, b11)
        level_mlb_coeff[(1, col)] = (0.0, 1.0, n11)
        level_mlb_coeff[(11, col)] = (a11, b11, n11)
        level_mlb_coeff[(14, col)] = (a14_to_1, b14_to_1, min(n14, n11))
        level_mlb_coeff[(16, col)] = (a16_to_1, b16_to_1, min(n16, n14, n11))
        coeff_rows.extend(
            [
                {
                    "src_level": 14,
                    "dst_level": 1,
                    "metric": col,
                    "a": a14_to_1,
                    "b": b14_to_1,
                    "n": min(n14, n11),
                    "min_src_tbf": 60,
                    "min_dst_tbf": 60,
                    "fit_type": "chained",
                },
                {
                    "src_level": 16,
                    "dst_level": 1,
                    "metric": col,
                    "a": a16_to_1,
                    "b": b16_to_1,
                    "n": min(n16, n14, n11),
                    "min_src_tbf": 60,
                    "min_dst_tbf": 60,
                    "fit_type": "chained",
                },
            ]
        )

    mlb_moments = moments[moments["level_id"] == 1].drop(columns=["level_id"]).copy()
    mlb_moments = mlb_moments.rename(
        columns={
            f"{col}__mean": f"{col}__mlb_mean"
            for col in metric_cols
            if f"{col}__mean" in mlb_moments.columns
        }
        | {
            f"{col}__std": f"{col}__mlb_std"
            for col in metric_cols
            if f"{col}__std" in mlb_moments.columns
        }
    )

    out = base_df.copy()
    out = out.merge(moments, on=["season", "level_id"], how="left")
    out = out.merge(mlb_moments, on=["season"], how="left")

    for col in metric_cols:
        src_mean_col = f"{col}__mean"
        src_std_col = f"{col}__std"
        mlb_mean_col = f"{col}__mlb_mean"
        mlb_std_col = f"{col}__mlb_std"
        if (
            col not in out.columns
            or src_mean_col not in out.columns
            or src_std_col not in out.columns
            or mlb_mean_col not in out.columns
            or mlb_std_col not in out.columns
        ):
            continue
        src_z = (out[col] - out[src_mean_col]) / out[src_std_col].replace(0, np.nan)
        a_map = {
            level_id: level_mlb_coeff.get((level_id, col), (prior_a, prior_b, 0))[0]
            for level_id in LEVEL_LABELS
        }
        b_map = {
            level_id: level_mlb_coeff.get((level_id, col), (prior_a, prior_b, 0))[1]
            for level_id in LEVEL_LABELS
        }
        pred_z = out["level_id"].map(a_map) + (out["level_id"].map(b_map) * src_z)
        pred = out[mlb_mean_col] + (pred_z * out[mlb_std_col])
        mlb_mask = out["level_id"] == 1
        non_mlb_mask = ~mlb_mask
        direction = PITCHER_MLB_DIRECTION_MAP.get(col)
        if direction == "up":
            pred.loc[non_mlb_mask] = np.maximum(
                pred.loc[non_mlb_mask],
                out.loc[non_mlb_mask, col],
            )
            shift = (
                (out[src_mean_col] - out[mlb_mean_col]).abs()
                * PITCHER_MLB_MIN_SHIFT_SCALE
            ).fillna(0.0)
            pred.loc[non_mlb_mask] = np.maximum(
                pred.loc[non_mlb_mask],
                out.loc[non_mlb_mask, col] + shift.loc[non_mlb_mask],
            )
        elif direction == "down":
            pred.loc[non_mlb_mask] = np.minimum(
                pred.loc[non_mlb_mask],
                out.loc[non_mlb_mask, col],
            )
            shift = (
                (out[src_mean_col] - out[mlb_mean_col]).abs()
                * PITCHER_MLB_MIN_SHIFT_SCALE
            ).fillna(0.0)
            pred.loc[non_mlb_mask] = np.minimum(
                pred.loc[non_mlb_mask],
                out.loc[non_mlb_mask, col] - shift.loc[non_mlb_mask],
            )
        pred.loc[mlb_mask] = out.loc[mlb_mask, col]
        out[f"{col}_mlb_eq"] = pred

    for col in PITCHER_MLB_PASS_THROUGH_COLS:
        if col in out.columns:
            out[f"{col}_mlb_eq"] = out[col]

    helper_cols: list[str] = []
    for col in metric_cols:
        helper_cols.extend(
            [f"{col}__mean", f"{col}__std", f"{col}__mlb_mean", f"{col}__mlb_std"]
        )
    out = out.drop(columns=[col for col in helper_cols if col in out.columns])

    coeff_df = pd.DataFrame(coeff_rows)
    all_mlb_eq_metrics = metric_cols + [
        col for col in PITCHER_MLB_PASS_THROUGH_COLS if f"{col}_mlb_eq" in out.columns
    ]
    all_mlb_eq_metrics = list(dict.fromkeys(all_mlb_eq_metrics))
    return out, coeff_df, all_mlb_eq_metrics


damage_df = _normalize_team_col(damage_df, "hitting_code")
damage_df = _normalize_la_cols(damage_df)
hitter_pct = _normalize_team_col(hitter_pct, "hitting_code")
hitter_pct = _normalize_la_cols(hitter_pct)
hitter_splits_df = _normalize_team_col(hitter_splits_df, "hitting_code")
hitter_splits_df = _normalize_la_cols(hitter_splits_df)
hitter_splits_df = _normalize_split_cols(hitter_splits_df)
pitcher_df = _normalize_team_col(pitcher_df, "pitching_code")
pitcher_df = _normalize_la_cols(pitcher_df)
pitcher_pct = _normalize_team_col(pitcher_pct, "pitching_code")
pitcher_splits_df = _normalize_team_col(pitcher_splits_df, "pitching_code")
pitcher_splits_df = _normalize_la_cols(pitcher_splits_df)
pitcher_splits_df = _normalize_split_cols(pitcher_splits_df)
pitch_types = _normalize_team_col(pitch_types, "pitching_code")
pitch_types_pct = _normalize_team_col(pitch_types_pct, "pitching_code")
pitch_type_splits_df = _normalize_team_col(pitch_type_splits_df, "pitching_code")
pitch_type_splits_df = _normalize_split_cols(pitch_type_splits_df)
league_pitch_types = _normalize_split_cols(league_pitch_types)
team_damage = _normalize_la_cols(team_damage)
team_stuff = _normalize_la_cols(team_stuff)

if (
    not pitch_types.empty
    and "pitch_group" not in pitch_types.columns
    and "pitch_tag" in pitch_types.columns
):
    pitch_types = pitch_types.assign(
        pitch_group=pitch_types["pitch_tag"].map(
            lambda tag: (
                "FA"
                if tag in {"FA", "HC", "SI"}
                else (
                    "BR"
                    if tag in {"SL", "SW", "CU"}
                    else "OFF" if tag in {"CH", "FS"} else "OTHER"
                )
            )
        )
    )

if (
    not pitch_type_splits_df.empty
    and "pitch_group" not in pitch_type_splits_df.columns
    and "pitch_tag" in pitch_type_splits_df.columns
):
    pitch_type_splits_df = pitch_type_splits_df.assign(
        pitch_group=pitch_type_splits_df["pitch_tag"].map(
            lambda tag: (
                "FA"
                if tag in {"FA", "HC", "SI"}
                else (
                    "BR"
                    if tag in {"SL", "SW", "CU"}
                    else "OFF" if tag in {"CH", "FS"} else "OTHER"
                )
            )
        )
    )

hitters_reg_df = _merge_regressed(
    damage_df,
    hitters_regressed,
    ["batter_mlbid", "hitter_name", "season", "level_id"],
)
pitchers_reg_df = _merge_regressed(
    pitcher_df,
    pitchers_regressed,
    ["pitcher_mlbid", "name", "season", "level_id", "pitcher_hand"],
)
pitch_types_reg_df = _merge_regressed(
    pitch_types,
    pitch_types_regressed,
    ["pitcher_mlbid", "name", "pitcher_hand", "season", "level_id", "pitch_tag"],
)
hitters_mlb_eq_df, hitter_mlb_eq_coeffs, hitter_mlb_eq_metrics = (
    _build_hitter_mlb_equivalencies(hitters_reg_df)
)
pitchers_mlb_eq_df, pitcher_mlb_eq_coeffs, pitcher_mlb_eq_metrics = (
    _build_pitcher_mlb_equivalencies(pitchers_reg_df)
)


# =============================================================================
# PAGE FUNCTIONS
# =============================================================================


def home_page():
    """Welcome/Home page"""
    st.title("The App & New Features")

    st.markdown(
        """
Welcome! Here you will find metrics I've developed for isolating and analyzing
the core skills that define hitters & pitchers at a player and team level. I made frequent use of these statistics in my player analysis work at BaseballProspectus dot com 
(https://www.baseballprospectus.com/author/ringtheodubel/) and for developing my fantasy strategies. 

You may recognize some of these from my Shiny app (https://therealestmuto.shinyapps.io/Damage/) but these have been updated with data from 2015-2025 and are slightly more
accurate and interpretable from their prior versions. SEAGER has a higher average total while Damage is lower, while the pitch metrics have been
converted to an overall pitch grade using the 20-80 scale familiar to baseball fans and applied within pitch types.

Each page contains some new statistics to go along with those you may already be familiar with from the other app, and each page can be filtered by logical conditions in the column filters dropdown.

Each page also features the ability to create a 2D visualization of the data, and the tables have conditional formatting similar to 
what you'll find on BaseballSavant, except in this case it's green=better and red=worse.

I hope you find everything useful!

-Robert Orr (https://twitter.com/NotTheBobbyOrr or https://bsky.app/profile/notthebobbyorr.bsky.social)
"""
    )

    st.markdown("---")
    st.subheader("The Pages")
    st.markdown(
        """
Navigate via the sidebar to explore the pages available:

The Auto Regressed (AR) pages contain the same information as the Individual Stats pages but have been 
stabilized for smaller samples to make players comparable across different seasons and playing time.

The Comps pages allow you to see similar player-seasons based on the same stats you'll find on the Stats and AR pages. The criteria used to make the comparison is customizable via the Similarity Score Columns area, where each metric available in the dataset can be included or excluded from the similarity calculation. The Similarity Score is displayed as a percentile compared to all other player-seasons in the dataset.

The Splits pages contain breakdowns by platoon matchup (vL/vR), home/away, 1st half/2nd half, and by month.

There are glossaries containing explanations for each statistic you may not recognize.

Pitch level comps are in the works and will be added soon, and I hope to have my own skill projections up before the 2026 season begins!
"""
    )
    st.write(f"Last Update: {pd.Timestamp.today().date()}")


# =============================================================================
# HITTERS PAGES
# =============================================================================


def hitter_individual_stats():
    """Hitters - Individual Stats page"""
    st.title("Individual Hitter Stats")

    if damage_df.empty:
        st.info("Missing damage_pos_2015_2025.csv")
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
            player_options, player_name_map = player_id_options(
                damage_df, "batter_mlbid", "hitter_name"
            )
            player = st.multiselect(
                "Select Player",
                player_options,
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
            df = damage_df.copy()
            df = df[df["level_id"].isin(level_map[level])]
            df = filter_by_values(df, "season", season)
            df = filter_by_team_token(df, "hitting_code", team)
            df = filter_by_positions(df, position)
            df = filter_by_values(df, "batter_mlbid", player)
            df = df.assign(__season=df["season"], __level=df["level_id"])

            if value_type == "PA":
                df = numeric_filter(df, "PA", min_value)
            else:
                df = numeric_filter(df, "bbe", min_value)

            columns = [
                "hitter_name",
                "batter_mlbid",
                "hitting_code",
                "season",
                "PA",
                "bbe",
                "damage_rate",
                "EV90th",
                "max_EV",
                "pull_FB_pct",
                "LA_gte_20",
                "LA_lte_0",
                "SEAGER",
                "selection_skill",
                "hittable_pitches_taken",
                "chase",
                "z_con",
                "secondary_whiff_pct",
                "whiffs_vs_95",
                "contact_vs_avg",
                "__season",
                "__level",
            ]
            df = df[[col for col in columns if col in df.columns]].copy()
            rename_map = {
                "hitter_name": "Name",
                "batter_mlbid": "Player ID",
                "hitting_code": "Team",
                "season": "Season",
                "bbe": "BBE",
                "damage_rate": "Damage/BBE (%)",
                "EV90th": "90th Pctile EV",
                "max_EV": "Max EV",
                "pull_FB_pct": "Pulled FB (%)",
                "LA_gte_20": "LA>=20%",
                "LA_lte_0": "LA<=0%",
                "selection_skill": "Selectivity (%)",
                "hittable_pitches_taken": "Hittable Pitch Take (%)",
                "chase": "Chase (%)",
                "z_con": "Z-Contact (%)",
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
            player_options, player_name_map = player_id_options(
                hitter_pct, "batter_mlbid", "hitter_name"
            )
            player = st.multiselect(
                "Select Player",
                player_options,
                default=["All"],
                format_func=lambda v: (
                    "All"
                    if v == "All"
                    else f"{player_name_map.get(v, 'Unknown')} ({int(v)})"
                ),
                key="hitter_pct_player",
            )
        with right:
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
            df = filter_by_team_token(df, "hitting_code", team)
            df = filter_by_positions(df, position)
            df = filter_by_values(df, "batter_mlbid", player)

            if value_type == "PA":
                df = numeric_filter(df, "PA", min_value)
            else:
                df = numeric_filter(df, "bbe", min_value)

            columns = [
                "hitter_name",
                "batter_mlbid",
                "hitting_code",
                "season",
                "SEAGER_pctile",
                "selection_skill_pctile",
                "hittable_pitches_taken_pctile",
                "damage_rate_pctile",
                "EV90th_pctile",
                "max_EV_pctile",
                "pull_FB_pct_pctile",
                "chase_pctile",
                "z_con_pctile",
                "secondary_whiff_pct_pctile",
                "whiffs_vs_95_pctile",
                "contact_vs_avg_pctile",
                "__season",
                "__level",
            ]
            df = df.assign(__season=df["season"], __level=df["level_id"])
            df = df[[col for col in columns if col in df.columns]].copy()
            rename_map = {
                "hitter_name": "Name",
                "batter_mlbid": "Player ID",
                "hitting_code": "Team",
                "season": "Season",
                "SEAGER_pctile": "SEAGER",
                "selection_skill_pctile": "Selection Skill",
                "hittable_pitches_taken_pctile": "Hittable Pitch Take",
                "damage_rate_pctile": "Damage Rate",
                "EV90th_pctile": "90th Pctile EV",
                "max_EV_pctile": "Max EV",
                "pull_FB_pct_pctile": "Pulled FB",
                "chase_pctile": "Chase",
                "z_con_pctile": "Z-Contact",
                "secondary_whiff_pct_pctile": "Whiff vs Secondaries",
                "whiffs_vs_95_pctile": "Whiff vs 95+",
                "contact_vs_avg_pctile": "Contact Over Expected",
            }
            df = df.rename(columns=rename_map)
            df = df.sort_values(by="Damage Rate", ascending=False)
            # For percentiles, reverse color on bad stats (higher pctile in bad stat = worse)
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

    player_options, player_name_map = player_id_options(
        season_df, "batter_mlbid", "hitter_name"
    )
    player_values = [opt for opt in player_options if opt != "All"]
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
        "batter_mlbid",
        "pitcher_mlbid",
        "level_id",
        "game_pk",
        "PA",
        "IP",
        "TBF",
        "GS",
        "pitches",
        "pitches_n",
        "pitches_num",
        "pitches_den",
        "bbe",
        "season",
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
        "hitter_name": "Name",
        "hitting_code": "Team",
        "season": "Season",
        "bbe": "BBE",
        "similarity_score": "Similarity (0-100)",
    }
    display_cols = [
        "hitter_name",
        "hitting_code",
        "season",
        "PA",
        "bbe",
        "similarity_score",
        *feature_cols,
        "__season",
        "__level",
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
        "hitter_name",
        "hitting_code",
        "season",
        "PA",
        "bbe",
        *list(dict.fromkeys(default_feature_cols + feature_cols)),
        "__season",
        "__level",
    ]
    stats_df = stats_df[
        [col for col in stats_columns if col in stats_df.columns]
    ].rename(columns={**base_rename, **similarity_labels})
    stats_df = stats_df.loc[:, ~stats_df.columns.duplicated()]

    target_cols = [
        "hitter_name",
        "hitting_code",
        "season",
        "PA",
        "bbe",
        *list(dict.fromkeys(default_feature_cols + feature_cols)),
        "__season",
        "__level",
    ]
    target_df = player_df.assign(
        __season=player_df["season"], __level=player_df["level_id"]
    )
    target_df = target_df[
        [col for col in target_cols if col in target_df.columns]
    ].copy()
    if use_mlb_eq and "__level" in target_df.columns:
        # MLB-equivalent stats should be color-scaled against MLB season baselines.
        target_df["__level"] = 1
    target_df = target_df.rename(columns={**base_rename, **similarity_labels})
    target_df = target_df.loc[:, ~target_df.columns.duplicated()]

    reverse_hitters = HIGHER_IS_WORSE_COLS | {
        "LA<=0%",
        "Chase (%)",
        "Swing Length",
        "Attack Angle",
        "VBA",
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
        "Level",
        level_options,
        index=0,
        key="hitter_mlb_eq_level",
    )
    level_map = {
        "All": [1, 11, 14, 16],
        "MLB": [1],
        "Triple-A": [11],
        "Low-A": [14],
        "Low Minors": [16],
    }
    view = view[view["level_id"].isin(level_map[level_choice])]

    team = st.selectbox(
        "Team",
        team_options(view, "hitting_code"),
        index=0,
        key="hitter_mlb_eq_team",
    )
    view = filter_by_team_token(view, "hitting_code", team)
    position = st.multiselect(
        "Position",
        position_options(view),
        default=["All"],
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
        "Metric Columns",
        options=metric_base_cols,
        default=default_metrics,
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
        "hitter_name",
        "batter_mlbid",
        "hitting_code",
        "season",
        "Level",
        "PA",
        "bbe",
        *metric_cols,
        "__season",
        "__level",
    ]
    show_cols = [col for col in show_cols if col in table_df.columns]
    table_df = table_df[show_cols].copy()

    rename_map = {
        "hitter_name": "Name",
        "batter_mlbid": "Player ID",
        "hitting_code": "Team",
        "season": "Season",
        "bbe": "BBE",
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
                    "metric": "Metric",
                    "from_level": "From",
                    "to_level": "To",
                    "a": "Intercept (a)",
                    "b": "Rate (b)",
                    "n": "Sample",
                    "fit_type": "Type",
                    "min_src_pa": "Min PA (From)",
                    "min_dst_pa": "Min PA (To)",
                }
            )
            coeff_cols = [
                "Metric",
                "From",
                "To",
                "Type",
                "Intercept (a)",
                "Rate (b)",
                "Sample",
                "Min PA (From)",
                "Min PA (To)",
            ]
            coeff_df = coeff_df[[col for col in coeff_cols if col in coeff_df.columns]]
            render_table(coeff_df, show_controls=False, round_decimals=2)


def hitter_ar():
    """Hitters - Auto Regressed page"""
    st.title("Hitters - Auto Regressed")

    if hitters_reg_df.empty:
        st.info("Missing hitters_regressed.csv or damage_pos_2015_2025.csv")
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
            player_options, player_name_map = player_id_options(
                hitters_reg_df, "batter_mlbid", "hitter_name"
            )
            player = st.multiselect(
                "Select Player",
                player_options,
                default=["All"],
                format_func=lambda v: (
                    "All"
                    if v == "All"
                    else f"{player_name_map.get(v, 'Unknown')} ({int(v)})"
                ),
                key="hitter_ar_player",
            )
        with right:
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
            df = filter_by_team_token(df, "hitting_code", team)
            df = filter_by_positions(df, position)
            df = filter_by_values(df, "batter_mlbid", player)

            if value_type == "PA":
                df = numeric_filter(df, "PA", min_value)
            else:
                df = numeric_filter(df, "bbe", min_value)

            columns = [
                "hitter_name",
                "batter_mlbid",
                "hitting_code",
                "season",
                "PA",
                "bbe",
                "damage_rate_reg",
                "EV90th_reg",
                "max_EV_reg",
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
                "__season",
                "__level",
            ]
            df = df.assign(__season=df["season"], __level=df["level_id"])
            df = df[[col for col in columns if col in df.columns]].copy()
            rename_map = {
                "hitter_name": "Name",
                "batter_mlbid": "Player ID",
                "hitting_code": "Team",
                "season": "Season",
                "bbe": "BBE",
                "damage_rate_reg": "Damage/BBE (%)",
                "EV90th_reg": "90th Pctile EV",
                "max_EV_reg": "Max EV",
                "pull_FB_pct_reg": "Pulled FB (%)",
                "LA_gte_20_reg": "LA>=20%",
                "LA_lte_0_reg": "LA<=0%",
                "SEAGER_reg": "SEAGER",
                "selection_skill_reg": "Selectivity (%)",
                "hittable_pitches_taken_reg": "Hittable Pitch Take (%)",
                "chase_reg": "Chase (%)",
                "z_con_reg": "Z-Contact (%)",
                "secondary_whiff_pct_reg": "Whiff vs. Secondaries (%)",
                "whiffs_vs_95_reg": "Whiff vs. 95+ (%)",
                "contact_vs_avg_reg": "Contact Over Expected (%)",
                "max_EV_reg": "Max EV",
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
    """Hitters - Splits page (placeholder)"""
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
                player_options, player_name_map = player_id_options(
                    split_df, "batter_mlbid", "hitter_name"
                )
                player = st.multiselect(
                    "Select Player",
                    player_options,
                    default=["All"],
                    format_func=lambda v: (
                        "All"
                        if v == "All"
                        else f"{player_name_map.get(v, 'Unknown')} ({int(v)})"
                    ),
                    key=f"hitter_splits_player_{idx}",
                )
            with right:
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
                    "hitter_name",
                    "batter_mlbid",
                    "hitting_code",
                    "season",
                    "split",
                    "PA",
                    "bbe",
                    "damage_rate",
                    "EV90th",
                    "max_EV",
                    "pull_FB_pct",
                    "LA_gte_20",
                    "LA_lte_0",
                    "SEAGER",
                    "selection_skill",
                    "hittable_pitches_taken",
                    "chase",
                    "z_con",
                    "secondary_whiff_pct",
                    "whiffs_vs_95",
                    "contact_vs_avg",
                    "__season",
                    "__level",
                ]
                df = df[[col for col in columns if col in df.columns]].copy()
                rename_map = {
                    "hitter_name": "Name",
                    "batter_mlbid": "Player ID",
                    "hitting_code": "Team",
                    "season": "Season",
                    "split": "Split",
                    "bbe": "BBE",
                    "damage_rate": "Damage/BBE (%)",
                    "EV90th": "90th Pctile EV",
                    "max_EV": "Max EV",
                    "pull_FB_pct": "Pulled FB (%)",
                    "LA_gte_20": "LA>=20%",
                    "LA_lte_0": "LA<=0%",
                    "selection_skill": "Selectivity (%)",
                    "hittable_pitches_taken": "Hittable Pitch Take (%)",
                    "chase": "Chase (%)",
                    "z_con": "Z-Contact (%)",
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


# =============================================================================
# PITCHERS PAGES
# =============================================================================


def pitcher_individual_stats():
    """Pitchers - Individual Stats page"""
    st.title("Individual Pitcher Stats")

    if pitcher_df.empty:
        st.info("Missing pitcher_stuff_new.csv")
    else:
        left, right = st.columns([1, 3])
        with left:
            level = st.selectbox(
                "Select Level",
                ["All", "MLB", "Triple-A", "Low-A", "Low Minors"],
                index=1,
                key="pitcher_stats_level",
            )
            season = st.multiselect(
                "Select Season",
                season_options(pitcher_df),
                default=(
                    [season_options(pitcher_df)[1]]
                    if len(season_options(pitcher_df)) > 1
                    else ["All"]
                ),
                key="pitcher_stats_season",
            )
            min_value = st.number_input(
                "Minimum Value",
                min_value=0,
                max_value=1000,
                value=100,
                step=1,
                key="pitcher_stats_min_value",
            )
            filter_type = st.selectbox(
                "Filter By", ["IP", "TBF", "GS"], index=1, key="pitcher_stats_filter_type"
            )
            team = st.selectbox(
                "Select Team",
                team_options(pitcher_df, "pitching_code"),
                index=0,
                key="pitcher_stats_team",
            )
            player_options, player_name_map = player_id_options(
                pitcher_df, "pitcher_mlbid", "name"
            )
            player = st.multiselect(
                "Select Player",
                player_options,
                default=["All"],
                format_func=lambda v: (
                    "All"
                    if v == "All"
                    else f"{player_name_map.get(v, 'Unknown')} ({int(v)})"
                ),
                key="pitcher_stats_player",
            )
        with right:
            level_map = {
                "All": [1, 11, 14, 16],
                "MLB": [1],
                "Triple-A": [11],
                "Low-A": [14],
                "Low Minors": [16],
            }
            base_stats = pitcher_df.copy()
            base_stats = base_stats.assign(
                __season=base_stats["season"], __level=base_stats["level_id"]
            )
            df = pitcher_df.copy()
            df = df[df["level_id"].isin(level_map[level])]
            df = filter_by_values(df, "season", season)
            df = filter_by_team_token(df, "pitching_code", team)
            df = filter_by_values(df, "pitcher_mlbid", player)
            df = df.assign(__season=df["season"], __level=df["level_id"])

            df = pitcher_workload_filter(df, filter_type, min_value)

            columns = [
                "name",
                "pitcher_mlbid",
                "pitching_code",
                "season",
                "TBF",
                "IP",
                "GS",
                "stuff",
                "fastball_velo",
                "max_velo",
                "fastball_vaa",
                "FA_pct",
                "BB_rpm",
                "SwStr",
                "Zone",
                "Ball_pct",
                "Z_Contact",
                "Chase",
                "CSW",
                "LA_lte_0",
                "rel_z",
                "rel_x",
                "ext",
                "__season",
                "__level",
            ]
            df = df[[col for col in columns if col in df.columns]].copy()
            # Round BB_rpm and stuff to integers
            if "BB_rpm" in df.columns:
                df["BB_rpm"] = df["BB_rpm"].round(0)
            if "stuff" in df.columns:
                df["stuff"] = df["stuff"].round(0)
            rename_map = {
                "name": "Name",
                "pitcher_mlbid": "Player ID",
                "pitching_code": "Team",
                "season": "Season",
                "GS": "GS",
                "stuff": "Pitch Grade",
                "fastball_velo": "FA mph",
                "max_velo": "Max FA mph",
                "fastball_vaa": "FA VAA",
                "FA_pct": "FA Usage (%)",
                "BB_rpm": "BB Spin",
                "SwStr": "SwStr (%)",
                "Zone": "Zone (%)",
                "Ball_pct": "Ball (%)",
                "Z_Contact": "Z-Contact (%)",
                "Chase": "Chase (%)",
                "CSW": "CSW (%)",
                "LA_lte_0": "LA<=0%",
                "rel_z": "Vertical Release (ft.)",
                "rel_x": "Horizontal Release (ft.)",
                "ext": "Extension (ft.)",
            }
            df = df.rename(columns=rename_map)
            df = df.sort_values(by="Pitch Grade", ascending=False)
            stats_df = base_stats[
                [col for col in columns if col in base_stats.columns]
            ].rename(columns=rename_map)
            render_table(
                df,
                reverse_cols={"Ball (%)", "FA VAA", "Z-Contact (%)"},
                group_cols=["__season", "__level"],
                stats_df=stats_df,
                abs_cols=ABS_GRADIENT_COLS_PITCHERS,
            )
            download_button(df, "pitchers", "pitchers_download")


def pitcher_percentiles():
    """Pitchers - Percentiles page"""
    st.title("Pitcher Percentiles")

    if pitcher_pct.empty:
        st.info("Missing pitcher_pctiles.csv")
    else:
        left, right = st.columns([1, 3])
        with left:
            level = st.selectbox(
                "Select Level",
                ["All", "MLB", "Triple-A", "Low-A", "Low Minors"],
                index=1,
                key="pitcher_pct_level",
            )
            season = st.multiselect(
                "Select Season",
                season_options(pitcher_pct),
                default=(
                    [season_options(pitcher_pct)[1]]
                    if len(season_options(pitcher_pct)) > 1
                    else ["All"]
                ),
                key="pitcher_pct_season",
            )
            min_value = st.number_input(
                "Minimum Value",
                min_value=0,
                max_value=1000,
                value=100,
                step=1,
                key="pitcher_pct_min_value",
            )
            filter_type = st.selectbox(
                "Filter By",
                ["TBF", "IP", "GS"],
                index=0,
                key="pitcher_pct_filter_type",
            )
            team = st.selectbox(
                "Select Team",
                team_options(pitcher_pct, "pitching_code"),
                index=0,
                key="pitcher_pct_team",
            )
            player_options, player_name_map = player_id_options(
                pitcher_pct, "pitcher_mlbid", "name"
            )
            player = st.multiselect(
                "Select Player",
                player_options,
                default=["All"],
                format_func=lambda v: (
                    "All"
                    if v == "All"
                    else f"{player_name_map.get(v, 'Unknown')} ({int(v)})"
                ),
                key="pitcher_pct_player",
            )
        with right:
            level_map = {
                "All": [1, 11, 14, 16],
                "MLB": [1],
                "Triple-A": [11],
                "Low-A": [14],
                "Low Minors": [16],
            }
            df = pitcher_pct.copy()
            df = df[df["level_id"].isin(level_map[level])]
            df = filter_by_values(df, "season", season)
            df = filter_by_team_token(df, "pitching_code", team)
            df = filter_by_values(df, "pitcher_mlbid", player)
            df = pitcher_workload_filter(df, filter_type, min_value)

            columns = [
                "name",
                "pitcher_mlbid",
                "pitching_code",
                "season",
                "GS",
                "stuff_pctile",
                "fastball_velo_pctile",
                "max_velo_pctile",
                "fastball_vaa_pctile",
                "SwStr_pctile",
                "Ball_pct_pctile",
                "Z_Contact_pctile",
                "Chase_pctile",
                "CSW_pctile",
                "rel_z_pctile",
                "rel_x_pctile",
                "ext_pctile",
                "__season",
                "__level",
            ]
            df = df.assign(__season=df["season"], __level=df["level_id"])
            df = df[[col for col in columns if col in df.columns]].copy()
            rename_map = {
                "name": "Name",
                "pitcher_mlbid": "Player ID",
                "pitching_code": "Team",
                "season": "Season",
                "GS": "GS",
                "stuff_pctile": "Pitch Grade Pctile",
                "fastball_velo_pctile": "Avg FA mph",
                "max_velo_pctile": "Max FA mph",
                "fastball_vaa_pctile": "FA VAA",
                "SwStr_pctile": "SwStr (%)",
                "Ball_pct_pctile": "Ball (%)",
                "Z_Contact_pctile": "Z-Contact (%)",
                "Chase_pctile": "Chase (%)",
                "CSW_pctile": "CSW (%)",
                "rel_z_pctile": "Vertical Release (ft.)",
                "rel_x_pctile": "Horizontal Release (ft.)",
                "ext_pctile": "Extension (ft.)",
            }
            df = df.rename(columns=rename_map)
            df = df.sort_values(by="Pitch Grade Pctile", ascending=False)
            render_table(
                df,
                reverse_cols={"FA VAA", "Ball (%)", "Z-Contact (%)"},
                abs_cols=ABS_GRADIENT_COLS_PITCHERS,
            )
            download_button(df, "pitcher_percentiles", "pitcher_pct_download")


def pitcher_comps():
    """Pitchers - Comparisons page"""
    st.title("Pitcher Comparisons (Auto-Regressed)")

    if pitchers_reg_df.empty:
        st.info("Missing pitchers_regressed.csv or pitcher_stuff_new.csv")
        return

    use_mlb_eq = st.toggle(
        "Use MLB-equivalent translated stats",
        value=False,
        key="pitcher_comps_use_mlb_eq",
        help=(
            "Use intra+inter-season level translations to compare non-MLB seasons "
            "against MLB seasons in MLB-equivalent space."
        ),
    )
    comp_df = pitchers_reg_df.copy()
    if use_mlb_eq:
        if pitchers_mlb_eq_df.empty:
            st.info(
                "MLB-equivalent translation table is unavailable; using raw AR stats."
            )
            use_mlb_eq = False
        else:
            comp_df = pitchers_mlb_eq_df.copy()

    if use_mlb_eq:
        player_pool = comp_df[(comp_df["TBF"] >= 20)].copy()
        eligible_all = comp_df[
            (comp_df["level_id"] == 1) & (comp_df["TBF"] >= 200)
        ].copy()
        if player_pool.empty:
            st.info("No eligible pitcher seasons (min 20 TBF).")
            return
        target_levels = sorted(player_pool["level_id"].dropna().unique().tolist())
        if not target_levels:
            st.info("No target levels available.")
            return
        target_level = st.selectbox(
            "Target Level",
            target_levels,
            index=0,
            key="pitcher_comps_target_level",
            format_func=lambda v: LEVEL_LABELS.get(int(v), str(int(v))),
        )
        player_pool = player_pool[player_pool["level_id"] == target_level]
    else:
        player_pool = comp_df[(comp_df["level_id"] == 1) & (comp_df["IP"] >= 5)].copy()
        eligible_all = comp_df[
            (comp_df["level_id"] == 1) & (comp_df["IP"] >= 50)
        ].copy()

    if player_pool.empty:
        if use_mlb_eq:
            st.info("No eligible pitcher seasons for that level (min 20 TBF).")
        else:
            st.info("No eligible MLB pitcher seasons (min 5 IP).")
        return
    if eligible_all.empty:
        if use_mlb_eq:
            st.info("No eligible MLB comparison seasons (min 200 TBF).")
        else:
            st.info("No eligible MLB comparison seasons (min 50 IP).")
        return

    seasons = season_options(player_pool, "season")[1:]
    if not seasons:
        st.info("No seasons available for this view.")
        return
    season_choice = st.selectbox("Season", seasons, index=0, key="pitcher_comps_season")
    season_df = player_pool[player_pool["season"] == season_choice]
    if season_df.empty:
        st.info("No player rows for this season selection.")
        return

    player_options, player_name_map = player_id_options(
        season_df, "pitcher_mlbid", "name"
    )
    player_values = [opt for opt in player_options if opt != "All"]
    if not player_values:
        st.info("No players available for this filter.")
        return
    player_choice = st.selectbox(
        "Player",
        player_values,
        index=0,
        format_func=lambda v: f"{player_name_map.get(v, 'Unknown')} ({int(v)})",
        key="pitcher_comps_player",
    )
    player_df = season_df[season_df["pitcher_mlbid"] == player_choice]
    teams = team_options(player_df, "pitching_code")
    team_choice = st.selectbox("Team", teams, index=0, key="pitcher_comps_team")
    if team_choice and team_choice != "All":
        filtered_player_df = filter_by_team_token(
            player_df, "pitching_code", team_choice
        )
        if filtered_player_df.empty:
            st.warning(
                "No rows for that team at this level/season; using all team rows for selection."
            )
        else:
            player_df = filtered_player_df

    metric_suffix = "_mlb_eq" if use_mlb_eq else ""
    default_feature_cols = [
        f"{col}{metric_suffix}" for col in PITCHER_COMPS_BASE_FEATURE_COLS
    ]
    allowed_cols = list(
        dict.fromkeys(
            default_feature_cols
            + [f"{col}{metric_suffix}" for col in PITCHER_COMPS_EXTRA_FEATURE_COLS]
        )
    )
    display_map = _pitcher_display_map(include_mlb_eq=False)
    if use_mlb_eq:
        base_display_map = _pitcher_display_map(include_mlb_eq=False)
        for col, label in base_display_map.items():
            if (
                col
                in PITCHER_COMPS_BASE_FEATURE_COLS + PITCHER_COMPS_EXTRA_FEATURE_COLS
            ):
                display_map[f"{col}_mlb_eq"] = label

    exclude_cols = {
        "batter_mlbid",
        "pitcher_mlbid",
        "level_id",
        "game_pk",
        "PA",
        "IP",
        "TBF",
        "GS",
        "pitches",
        "pitches_n",
        "pitches_num",
        "pitches_den",
        "bbe",
        "season",
    }
    numeric_cols, similarity_labels = _similarity_choice_labels(
        eligible_all, display_map, exclude_cols
    )
    numeric_cols = [col for col in numeric_cols if col in allowed_cols]
    default_feature_cols = [col for col in default_feature_cols if col in numeric_cols]
    similarity_key = (
        "pitcher_comps_similarity_cols_mlb_eq"
        if use_mlb_eq
        else "pitcher_comps_similarity_cols_raw"
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
    eligible_comp = eligible_comp[~(eligible_comp["pitcher_mlbid"] == player_choice)]
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
        "name": "Name",
        "pitching_code": "Team",
        "season": "Season",
        "similarity_score": "Similarity (0-100)",
    }
    display_cols = [
        "name",
        "pitching_code",
        "season",
        "TBF",
        "IP",
        "GS",
        "similarity_score",
        *feature_cols,
        "__season",
        "__level",
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
        "name",
        "pitching_code",
        "season",
        "TBF",
        "IP",
        "GS",
        *list(dict.fromkeys(default_feature_cols + feature_cols)),
        "__season",
        "__level",
    ]
    stats_df = stats_df[
        [col for col in stats_columns if col in stats_df.columns]
    ].rename(columns={**base_rename, **similarity_labels})
    stats_df = stats_df.loc[:, ~stats_df.columns.duplicated()]

    target_cols = [
        "name",
        "pitching_code",
        "season",
        "TBF",
        "IP",
        "GS",
        *list(dict.fromkeys(default_feature_cols + feature_cols)),
        "__season",
        "__level",
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

    if use_mlb_eq:
        level_label = LEVEL_LABELS.get(int(target_level), str(int(target_level)))
        player_name = str(
            player_df["name"].iloc[0]
            if "name" in player_df.columns and not player_df.empty
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
        reverse_cols=PITCHER_REVERSE_DISPLAY_COLS,
        group_cols=["__season", "__level"],
        stats_df=stats_df,
        abs_cols=ABS_GRADIENT_COLS_PITCHERS,
        show_controls=False,
        hide_cols={"Team"},
    )
    if use_mlb_eq:
        st.caption(
            "Most similar MLB seasons by translated MLB-equivalent stats (TBF >= 200)"
        )
    else:
        st.caption("Most similar MLB seasons (IP >= 50)")
    render_table(
        df,
        reverse_cols=PITCHER_REVERSE_DISPLAY_COLS,
        group_cols=["__season", "__level"],
        stats_df=stats_df,
        abs_cols=ABS_GRADIENT_COLS_PITCHERS,
    )


def pitcher_mlb_equivalencies():
    """Pitchers - MLB equivalency translations (intra+inter-season chained)."""
    st.title("Pitcher MLB Equivalencies")

    if pitchers_mlb_eq_df.empty:
        st.info("MLB-equivalent table is unavailable.")
        return

    st.caption(
        "Intra+inter-season chained translations in season-adjusted z-score space "
        "(16->14->11->1)."
    )
    st.caption(
        "Pitch Grade, FA mph, Vertical Release, Horizontal Release, Extension, and "
        "Arm Angle are treated as MLB-equivalent pass-through metrics."
    )
    st.caption("Direct edge fits use TBF >= 60 at both source and destination levels.")

    view = pitchers_mlb_eq_df.copy()
    if "TBF" in view.columns:
        min_tbf = st.number_input(
            "Minimum TBF",
            min_value=0,
            max_value=2000,
            value=20,
            step=10,
            key="pitcher_mlb_eq_min_tbf",
        )
        view = view[view["TBF"] >= min_tbf]

    season_vals = season_options(view)
    season = st.selectbox(
        "Season",
        season_vals,
        index=(1 if len(season_vals) > 1 else 0),
        key="pitcher_mlb_eq_season",
    )
    view = filter_by_values(view, "season", season)

    level_options = ["All", "MLB", "Triple-A", "Low-A", "Low Minors"]
    level_choice = st.selectbox(
        "Level",
        level_options,
        index=0,
        key="pitcher_mlb_eq_level",
    )
    level_map = {
        "All": [1, 11, 14, 16],
        "MLB": [1],
        "Triple-A": [11],
        "Low-A": [14],
        "Low Minors": [16],
    }
    view = view[view["level_id"].isin(level_map[level_choice])]

    team = st.selectbox(
        "Team",
        team_options(view, "pitching_code"),
        index=0,
        key="pitcher_mlb_eq_team",
    )
    view = filter_by_team_token(view, "pitching_code", team)
    if view.empty:
        st.info("No rows after filtering.")
        return

    metric_base_cols = [
        col
        for col in PITCHER_COMPS_BASE_FEATURE_COLS + PITCHER_COMPS_EXTRA_FEATURE_COLS
        if col in view.columns and f"{col}_mlb_eq" in view.columns
    ]
    default_metrics = [
        col for col in PITCHER_COMPS_BASE_FEATURE_COLS if col in metric_base_cols
    ]
    selected_metrics = st.multiselect(
        "Metric Columns",
        options=metric_base_cols,
        default=default_metrics,
        key="pitcher_mlb_eq_metrics",
        format_func=lambda col: _pitcher_display_map().get(col, col),
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
        "name",
        "pitcher_mlbid",
        "pitching_code",
        "season",
        "Level",
        "TBF",
        "IP",
        "GS",
        *metric_cols,
        "__season",
        "__level",
    ]
    show_cols = [col for col in show_cols if col in table_df.columns]
    table_df = table_df[show_cols].copy()

    rename_map = {
        "name": "Name",
        "pitcher_mlbid": "Player ID",
        "pitching_code": "Team",
        "season": "Season",
    }
    display_map = _pitcher_display_map(include_mlb_eq=True)
    for col in selected_metrics:
        rename_map[col] = display_map.get(col, col)
        rename_map[f"{col}_mlb_eq"] = display_map.get(f"{col}_mlb_eq", f"{col} MLB Eq")
        rename_map[f"{col}_mlb_delta"] = (
            f"{display_map.get(f'{col}_mlb_eq', f'{col} MLB Eq')} Delta"
        )
    table_df = table_df.rename(columns=rename_map)

    reverse_cols = PITCHER_REVERSE_DISPLAY_COLS | {
        f"{name} MLB Eq" for name in PITCHER_REVERSE_DISPLAY_COLS
    }
    render_table(
        table_df,
        reverse_cols=reverse_cols,
        group_cols=["__season", "__level"],
        stats_df=table_df,
        abs_cols=ABS_GRADIENT_COLS_PITCHERS,
    )
    download_button(table_df, "pitcher_mlb_equivalencies", "pitcher_mlb_eq_download")

    if not pitcher_mlb_eq_coeffs.empty:
        with st.expander("Translation coefficients", expanded=False):
            coeff_df = pitcher_mlb_eq_coeffs.copy()
            coeff_df = coeff_df.assign(
                from_level=coeff_df["src_level"].map(
                    lambda v: LEVEL_LABELS.get(int(v), str(int(v)))
                ),
                to_level=coeff_df["dst_level"].map(
                    lambda v: LEVEL_LABELS.get(int(v), str(int(v)))
                ),
            )
            coeff_df["metric"] = coeff_df["metric"].map(
                lambda c: _pitcher_display_map().get(c, c)
            )
            coeff_df = coeff_df.rename(
                columns={
                    "metric": "Metric",
                    "from_level": "From",
                    "to_level": "To",
                    "a": "Intercept (a)",
                    "b": "Rate (b)",
                    "n": "Sample",
                    "fit_type": "Type",
                    "min_src_tbf": "Min TBF (From)",
                    "min_dst_tbf": "Min TBF (To)",
                }
            )
            coeff_cols = [
                "Metric",
                "From",
                "To",
                "Type",
                "Intercept (a)",
                "Rate (b)",
                "Sample",
                "Min TBF (From)",
                "Min TBF (To)",
            ]
            coeff_df = coeff_df[[col for col in coeff_cols if col in coeff_df.columns]]
            render_table(coeff_df, show_controls=False, round_decimals=2)


def pitcher_ar():
    """Pitchers - Auto Regressed page"""
    st.title("Pitchers - Auto Regressed")

    if pitchers_reg_df.empty:
        st.info("Missing pitchers_regressed.csv or pitcher_stuff_new.csv")
    else:
        left, right = st.columns([1, 3])
        with left:
            level = st.selectbox(
                "Select Level",
                ["All", "MLB", "Triple-A", "Low-A", "Low Minors"],
                index=1,
                key="pitcher_ar_level",
            )
            season = st.multiselect(
                "Select Season",
                season_options(pitchers_reg_df),
                default=(
                    [season_options(pitchers_reg_df)[1]]
                    if len(season_options(pitchers_reg_df)) > 1
                    else ["All"]
                ),
                key="pitcher_ar_season",
            )
            min_value = st.number_input(
                "Minimum Value",
                min_value=0,
                max_value=1000,
                value=100,
                step=1,
                key="pitcher_ar_min_value",
            )
            filter_type = st.selectbox(
                "Filter By", ["IP", "TBF", "GS"], index=1, key="pitcher_ar_filter_type"
            )
            team = st.selectbox(
                "Select Team",
                team_options(pitchers_reg_df, "pitching_code"),
                index=0,
                key="pitcher_ar_team",
            )
            player_options, player_name_map = player_id_options(
                pitchers_reg_df, "pitcher_mlbid", "name"
            )
            player = st.multiselect(
                "Select Player",
                player_options,
                default=["All"],
                format_func=lambda v: (
                    "All"
                    if v == "All"
                    else f"{player_name_map.get(v, 'Unknown')} ({int(v)})"
                ),
                key="pitcher_ar_player",
            )
        with right:
            level_map = {
                "All": [1, 11, 14, 16],
                "MLB": [1],
                "Triple-A": [11],
                "Low-A": [14],
                "Low Minors": [16],
            }
            base_stats = pitchers_reg_df.copy()
            base_stats = base_stats.assign(
                __season=base_stats["season"], __level=base_stats["level_id"]
            )
            df = pitchers_reg_df.copy()
            df = df[df["level_id"].isin(level_map[level])]
            df = filter_by_values(df, "season", season)
            df = filter_by_team_token(df, "pitching_code", team)
            df = filter_by_values(df, "pitcher_mlbid", player)
            df = df.assign(__season=df["season"], __level=df["level_id"])

            df = pitcher_workload_filter(df, filter_type, min_value)

            columns = [
                "name",
                "pitcher_mlbid",
                "pitching_code",
                "season",
                "TBF",
                "IP",
                "GS",
                "stuff",
                "fastball_velo_reg",
                "max_velo_reg",
                "fastball_vaa_reg",
                "FA_pct_reg",
                "BB_rpm_reg",
                "SwStr_reg",
                "Ball_pct_reg",
                "Z_Contact_reg",
                "Chase_reg",
                "CSW_reg",
                "LA_lte_0_reg",
                "rel_z_reg",
                "rel_x_reg",
                "ext_reg",
                "__season",
                "__level",
            ]
            df = df[[col for col in columns if col in df.columns]].copy()
            rename_map = {
                "name": "Name",
                "pitcher_mlbid": "Player ID",
                "pitching_code": "Team",
                "season": "Season",
                "GS": "GS",
                "stuff": "Pitch Grade",
                "fastball_velo_reg": "FA mph",
                "max_velo_reg": "Max FA mph",
                "fastball_vaa_reg": "FA VAA",
                "FA_pct_reg": "FA Usage (%)",
                "BB_rpm_reg": "BB Spin",
                "SwStr_reg": "SwStr (%)",
                "Ball_pct_reg": "Ball (%)",
                "Z_Contact_reg": "Z-Contact (%)",
                "Chase_reg": "Chase (%)",
                "CSW_reg": "CSW (%)",
                "LA_lte_0_reg": "LA<=0%",
                "rel_z_reg": "Vertical Release (ft.)",
                "rel_x_reg": "Horizontal Release (ft.)",
                "ext_reg": "Extension (ft.)",
            }
            df = df.rename(columns=rename_map)
            df = df.sort_values(by="Pitch Grade", ascending=False)
            stats_df = base_stats[
                [col for col in columns if col in base_stats.columns]
            ].rename(columns=rename_map)
            render_table(
                df,
                reverse_cols={"Ball (%)", "FA VAA", "Z-Contact (%)"},
                group_cols=["__season", "__level"],
                stats_df=stats_df,
                abs_cols=ABS_GRADIENT_COLS_PITCHERS,
            )
            download_button(df, "pitchers_ar", "pitchers_ar_download")


def pitcher_splits():
    """Pitchers - Splits page (placeholder)"""
    st.title("Pitcher Splits")

    if pitcher_splits_df.empty:
        st.info("Missing pitcher_splits.csv")
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
            split_df = pitcher_splits_df[
                pitcher_splits_df["split_type"] == split_type
            ].copy()
            if split_df.empty:
                available = sorted(
                    pitcher_splits_df["split_type"].dropna().unique().tolist()
                )
                st.info(f"No data for {tab_name}. Available split types: {available}")
                continue

            left, right = st.columns([1, 3])
            with left:
                level = st.selectbox(
                    "Select Level",
                    ["All", "MLB", "Triple-A", "Low-A", "Low Minors"],
                    index=1,
                    key=f"pitcher_splits_level_{idx}",
                )
                season = st.multiselect(
                    "Select Season",
                    season_options(split_df),
                    default=(
                        [season_options(split_df)[1]]
                        if len(season_options(split_df)) > 1
                        else ["All"]
                    ),
                    key=f"pitcher_splits_season_{idx}",
                )
                min_value = st.number_input(
                    "Minimum Value",
                    min_value=0,
                    max_value=1000,
                    value=100,
                    step=1,
                    key=f"pitcher_splits_min_value_{idx}",
                )
                filter_type = st.selectbox(
                    "Filter By",
                    ["IP", "TBF", "GS"],
                    index=1,
                    key=f"pitcher_splits_filter_type_{idx}",
                )
                split_choice = st.multiselect(
                    "Select Split",
                    ["All"] + sorted(split_df["split"].dropna().unique().tolist()),
                    default=["All"],
                    key=f"pitcher_splits_split_{idx}",
                )
                team = st.selectbox(
                    "Select Team",
                    team_options(split_df, "pitching_code"),
                    index=0,
                    key=f"pitcher_splits_team_{idx}",
                )
                player_options, player_name_map = player_id_options(
                    split_df, "pitcher_mlbid", "name"
                )
                player = st.multiselect(
                    "Select Player",
                    player_options,
                    default=["All"],
                    format_func=lambda v: (
                        "All"
                        if v == "All"
                        else f"{player_name_map.get(v, 'Unknown')} ({int(v)})"
                    ),
                    key=f"pitcher_splits_player_{idx}",
                )
            with right:
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
                df = filter_by_values(df, "split", split_choice)
                df = filter_by_team_token(df, "pitching_code", team)
                df = filter_by_values(df, "pitcher_mlbid", player)
                df = df.assign(__season=df["season"], __level=df["level_id"])

                df = pitcher_workload_filter(df, filter_type, min_value)

                columns = [
                    "name",
                    "pitcher_mlbid",
                    "split",
                    "pitching_code",
                    "season",
                    "TBF",
                    "IP",
                    "GS",
                    "stuff",
                    "fastball_velo",
                    "max_velo",
                    "fastball_vaa",
                    "FA_pct",
                    "BB_rpm",
                    "SwStr",
                    "Ball_pct",
                    "Z_Contact",
                    "Chase",
                    "CSW",
                    "LA_lte_0",
                    "rel_z",
                    "rel_x",
                    "ext",
                    "__season",
                    "__level",
                ]
                df = df[[col for col in columns if col in df.columns]].copy()
                if "BB_rpm" in df.columns:
                    df["BB_rpm"] = df["BB_rpm"].round(0)
                if "stuff" in df.columns:
                    df["stuff"] = df["stuff"].round(0)
                rename_map = {
                    "name": "Name",
                    "pitcher_mlbid": "Player ID",
                    "pitching_code": "Team",
                    "season": "Season",
                    "split": "Split",
                    "GS": "GS",
                    "stuff": "Pitch Grade",
                    "fastball_velo": "FA mph",
                    "max_velo": "Max FA mph",
                    "fastball_vaa": "FA VAA",
                    "FA_pct": "FA Usage (%)",
                    "BB_rpm": "BB Spin",
                    "SwStr": "SwStr (%)",
                    "Ball_pct": "Ball (%)",
                    "Z_Contact": "Z-Contact (%)",
                    "Chase": "Chase (%)",
                    "CSW": "CSW (%)",
                    "LA_lte_0": "LA<=0%",
                    "rel_z": "Vertical Release (ft.)",
                    "rel_x": "Horizontal Release (ft.)",
                    "ext": "Extension (ft.)",
                }
                df = df.rename(columns=rename_map)
                df = df.sort_values(by="Pitch Grade", ascending=False)
                stats_df = base_stats[
                    [col for col in columns if col in base_stats.columns]
                ].rename(columns=rename_map)
                render_table(
                    df,
                    reverse_cols={"Ball (%)", "FA VAA", "Z-Contact (%)"},
                    group_cols=["__season", "__level"],
                    stats_df=stats_df,
                    abs_cols=ABS_GRADIENT_COLS_PITCHERS,
                )
                download_button(
                    df,
                    f"pitcher_splits_{idx}",
                    f"pitcher_splits_download_{idx}",
                )


# =============================================================================
# INDIVIDUAL PITCHES PAGES
# =============================================================================


def pitch_shapes_outcomes():
    """Individual Pitches - Shapes and Outcomes page"""
    st.title("Individual Pitches - Shapes and Outcomes")

    if pitch_types.empty:
        st.info("Missing new_pitch_types.csv")
    else:
        left, right = st.columns([1, 3])
        with left:
            level = st.selectbox(
                "Select Level",
                ["All", "MLB", "Triple-A", "Low-A", "Low Minors"],
                index=1,
                key="pitch_shapes_level",
            )
            season = st.multiselect(
                "Select Season",
                season_options(pitch_types),
                default=(
                    [season_options(pitch_types)[1]]
                    if len(season_options(pitch_types)) > 1
                    else ["All"]
                ),
                key="pitch_shapes_season",
            )
            min_pitches = st.number_input(
                "Minimum # Pitches",
                min_value=0,
                max_value=1000,
                value=50,
                step=1,
                key="pitch_shapes_min_pitches",
            )
            team = st.selectbox(
                "Select Team",
                team_options(pitch_types, "pitching_code"),
                index=0,
                key="pitch_shapes_team",
            )
            pitcher_options, pitcher_name_map = player_id_options(
                pitch_types, "pitcher_mlbid", "name"
            )
            pitcher = st.multiselect(
                "Select Pitcher",
                pitcher_options,
                default=["All"],
                format_func=lambda v: (
                    "All"
                    if v == "All"
                    else f"{pitcher_name_map.get(v, 'Unknown')} ({int(v)})"
                ),
                key="pitch_shapes_pitcher",
            )
            pitch_group = st.multiselect(
                "Select Pitch Group",
                ["All"] + sorted(pitch_types["pitch_group"].dropna().unique().tolist()),
                default=["All"],
                key="pitch_shapes_pitch_group",
            )
            pitch_tag = st.multiselect(
                "Select Pitch Type",
                ["All"] + sorted(pitch_types["pitch_tag"].dropna().unique().tolist()),
                default=["All"],
                key="pitch_shapes_pitch_tag",
            )
        with right:
            level_map = {
                "All": [1, 11, 14, 16],
                "MLB": [1],
                "Triple-A": [11],
                "Low-A": [14],
                "Low Minors": [16],
            }
            base_stats = pitch_types.copy()
            base_stats = base_stats.assign(
                __season=base_stats["season"], __level=base_stats["level_id"]
            )
            df = pitch_types.copy()
            df = df[df["level_id"].isin(level_map[level])]
            df = filter_by_values(df, "season", season)
            df = filter_by_team_token(df, "pitching_code", team)
            df = filter_by_values(df, "pitcher_mlbid", pitcher)
            df = filter_by_values(df, "pitch_group", pitch_group)
            df = filter_by_values(df, "pitch_tag", pitch_tag)
            df = df[df["pitches"] >= min_pitches]
            df = df.assign(__season=df["season"], __level=df["level_id"])

            columns = [
                "name",
                "pitcher_mlbid",
                "pitching_code",
                "season",
                "pitch_tag",
                "pitches",
                "pct",
                "stuff",
                "velo",
                "max_velo",
                "vaa",
                "haa",
                "vbreak",
                "hbreak",
                "SwStr",
                "LA_lte_0",
                "Z_Contact",
                "Ball_pct",
                "Zone",
                "Chase",
                "CSW",
                "__season",
                "__level",
            ]
            df = df[[col for col in columns if col in df.columns]].copy()
            # Round stuff to integer
            if "stuff" in df.columns:
                df["stuff"] = df["stuff"].round(0)
            rename_map = {
                "name": "Name",
                "pitcher_mlbid": "Player ID",
                "pitching_code": "Team",
                "season": "Season",
                "pitch_tag": "Pitch Type",
                "pitches": "#",
                "pct": "Usage (%)",
                "stuff": "Pitch Grade",
                "velo": "Velo",
                "max_velo": "Max Velo",
                "vaa": "VAA",
                "haa": "HAA",
                "vbreak": "IVB (in.)",
                "hbreak": "HB (in.)",
                "CSW": "CSW (%)",
                "SwStr": "SwStr (%)",
                "LA_lte_0": "LA<=0%",
                "Z_Contact": "Z-Contact (%)",
                "Chase": "Chase (%)",
                "Zone": "Zone (%)",
                "Ball_pct": "Ball (%)",
            }
            df = df.rename(columns=rename_map)
            df = df.sort_values(by="Pitch Grade", ascending=False)
            stats_df = base_stats[
                [col for col in columns if col in base_stats.columns]
            ].rename(columns=rename_map)
            render_table(
                df,
                reverse_cols={"Ball (%)", "Z-Contact (%)", "VAA"},
                group_cols=["__season", "__level"],
                stats_df=stats_df,
                abs_cols=ABS_GRADIENT_COLS_PITCH_TYPES,
                label_cols=["Name", "Pitch Type", "Split", "split", "Split Type"],
            )
            download_button(df, "pitch_types", "pitch_types_download")


def pitch_ar():
    """Individual Pitches - Auto Regressed page"""
    st.title("Individual Pitches - Auto Regressed")

    if pitch_types_reg_df.empty:
        st.info("Missing pitch_types_regressed.csv or new_pitch_types.csv")
    else:
        left, right = st.columns([1, 3])
        with left:
            level = st.selectbox(
                "Select Level",
                ["All", "MLB", "Triple-A", "Low-A", "Low Minors"],
                index=1,
                key="pitch_ar_level",
            )
            season = st.multiselect(
                "Select Season",
                season_options(pitch_types_reg_df),
                default=(
                    [season_options(pitch_types_reg_df)[1]]
                    if len(season_options(pitch_types_reg_df)) > 1
                    else ["All"]
                ),
                key="pitch_ar_season",
            )
            min_pitches = st.number_input(
                "Minimum # Pitches",
                min_value=0,
                max_value=1000,
                value=50,
                step=1,
                key="pitch_ar_min_pitches",
            )
            team = st.selectbox(
                "Select Team",
                team_options(pitch_types_reg_df, "pitching_code"),
                index=0,
                key="pitch_ar_team",
            )
            pitcher_options, pitcher_name_map = player_id_options(
                pitch_types_reg_df, "pitcher_mlbid", "name"
            )
            pitcher = st.multiselect(
                "Select Pitcher",
                pitcher_options,
                default=["All"],
                format_func=lambda v: (
                    "All"
                    if v == "All"
                    else f"{pitcher_name_map.get(v, 'Unknown')} ({int(v)})"
                ),
                key="pitch_ar_pitcher",
            )
            pitch_group = st.multiselect(
                "Select Pitch Group",
                ["All"]
                + sorted(pitch_types_reg_df["pitch_group"].dropna().unique().tolist()),
                default=["All"],
                key="pitch_ar_pitch_group",
            )
            pitch_tag = st.multiselect(
                "Select Pitch Type",
                ["All"]
                + sorted(pitch_types_reg_df["pitch_tag"].dropna().unique().tolist()),
                default=["All"],
                key="pitch_ar_pitch_tag",
            )
        with right:
            level_map = {
                "All": [1, 11, 14, 16],
                "MLB": [1],
                "Triple-A": [11],
                "Low-A": [14],
                "Low Minors": [16],
            }
            base_stats = pitch_types_reg_df.copy()
            base_stats = base_stats.assign(
                __season=base_stats["season"], __level=base_stats["level_id"]
            )
            df = pitch_types_reg_df.copy()
            df = df[df["level_id"].isin(level_map[level])]
            df = filter_by_values(df, "season", season)
            df = filter_by_team_token(df, "pitching_code", team)
            df = filter_by_values(df, "pitcher_mlbid", pitcher)
            df = filter_by_values(df, "pitch_group", pitch_group)
            df = filter_by_values(df, "pitch_tag", pitch_tag)
            df = df[df["pitches"] >= min_pitches]
            df = df.assign(__season=df["season"], __level=df["level_id"])

            columns = [
                "name",
                "pitcher_mlbid",
                "pitching_code",
                "season",
                "pitch_tag",
                "pitches",
                "pct",
                "stuff",
                "velo_reg",
                "max_velo_reg",
                "vaa_reg",
                "haa_reg",
                "vbreak_reg",
                "hbreak_reg",
                "SwStr_reg",
                "LA_lte_0_reg",
                "Z_Contact_reg",
                "Ball_pct_reg",
                "Chase_reg",
                "CSW_reg",
                "__season",
                "__level",
            ]
            df = df[[col for col in columns if col in df.columns]].copy()
            rename_map = {
                "name": "Name",
                "pitcher_mlbid": "Player ID",
                "pitching_code": "Team",
                "season": "Season",
                "pitch_tag": "Pitch Type",
                "pitches": "#",
                "pct": "Usage (%)",
                "stuff": "Pitch Grade",
                "velo_reg": "Velo",
                "max_velo_reg": "Max Velo",
                "vaa_reg": "VAA",
                "haa_reg": "HAA",
                "vbreak_reg": "IVB (in.)",
                "hbreak_reg": "HB (in.)",
                "CSW_reg": "CSW (%)",
                "SwStr_reg": "SwStr (%)",
                "LA_lte_0_reg": "LA<=0%",
                "Z_Contact_reg": "Z-Contact (%)",
                "Chase_reg": "Chase (%)",
                "Ball_pct_reg": "Ball (%)",
            }
            df = df.rename(columns=rename_map)
            df = df.sort_values(by="Pitch Grade", ascending=False)
            stats_df = base_stats[
                [col for col in columns if col in base_stats.columns]
            ].rename(columns=rename_map)
            render_table(
                df,
                reverse_cols={"Ball (%)", "Z-Contact (%)", "VAA"},
                group_cols=["__season", "__level"],
                stats_df=stats_df,
                abs_cols=ABS_GRADIENT_COLS_PITCH_TYPES,
                label_cols=["Name", "Pitch Type", "Split", "split", "Split Type"],
            )
            download_button(df, "pitch_types_ar", "pitch_types_ar_download")


def pitch_percentiles():
    """Individual Pitches - Percentiles page"""
    st.title("Individual Pitches - Percentiles")

    if pitch_types_pct.empty:
        st.info("Missing pitch_types_pctiles.csv")
    else:
        left, right = st.columns([1, 3])
        with left:
            level = st.selectbox(
                "Select Level",
                ["All", "MLB", "Triple-A", "Low-A", "Low Minors"],
                index=1,
                key="pitch_pct_level",
            )
            season = st.multiselect(
                "Select Season",
                season_options(pitch_types_pct),
                default=(
                    [season_options(pitch_types_pct)[1]]
                    if len(season_options(pitch_types_pct)) > 1
                    else ["All"]
                ),
                key="pitch_pct_season",
            )
            min_pitches = st.number_input(
                "Minimum # Pitches",
                min_value=0,
                max_value=1000,
                value=50,
                step=1,
                key="pitch_pct_min_pitches",
            )
            team = st.selectbox(
                "Select Team",
                team_options(pitch_types_pct, "pitching_code"),
                index=0,
                key="pitch_pct_team",
            )
            pitcher_options, pitcher_name_map = player_id_options(
                pitch_types_pct, "pitcher_mlbid", "name"
            )
            pitcher = st.multiselect(
                "Select Pitcher",
                pitcher_options,
                default=["All"],
                format_func=lambda v: (
                    "All"
                    if v == "All"
                    else f"{pitcher_name_map.get(v, 'Unknown')} ({int(v)})"
                ),
                key="pitch_pct_pitcher",
            )
            pitch_tag = st.multiselect(
                "Select Pitch Type",
                ["All"]
                + sorted(pitch_types_pct["pitch_tag"].dropna().unique().tolist()),
                default=["All"],
                key="pitch_pct_pitch_tag",
            )
        with right:
            level_map = {
                "All": [1, 11, 14, 16],
                "MLB": [1],
                "Triple-A": [11],
                "Low-A": [14],
                "Low Minors": [16],
            }
            df = pitch_types_pct.copy()
            df = df[df["level_id"].isin(level_map[level])]
            df = filter_by_values(df, "season", season)
            df = filter_by_team_token(df, "pitching_code", team)
            df = filter_by_values(df, "pitcher_mlbid", pitcher)
            df = filter_by_values(df, "pitch_tag", pitch_tag)
            df = df[df["pitches"] >= min_pitches]

            columns = [
                "name",
                "pitcher_mlbid",
                "pitching_code",
                "season",
                "pitch_tag",
                "pct",
                "stuff_z",
                "stuff_pctile",
                "velo_pctile",
                "max_velo_pctile",
                "vaa_pctile",
                "haa_pctile",
                "vbreak_pctile",
                "hbreak_pctile",
                "SwStr_pctile",
                "LA_lte_0_pctile",
                "Ball_pct_pctile",
                "Z_Contact_pctile",
                "Chase_pctile",
                "CSW_pctile",
                "__season",
                "__level",
            ]
            df = df.assign(__season=df["season"], __level=df["level_id"])
            df = df[[col for col in columns if col in df.columns]].copy()
            rename_map = {
                "name": "Name",
                "pitcher_mlbid": "Player ID",
                "pitching_code": "Team",
                "season": "Season",
                "pitch_tag": "Pitch Type",
                "pct": "Usage (%)",
                "stuff_z": "Pitch Grade Z",
                "stuff_pctile": "Pitch Grade Pctile",
                "velo_pctile": "Velo",
                "max_velo_pctile": "Max Velo",
                "vaa_pctile": "VAA",
                "haa_pctile": "HAA",
                "vbreak_pctile": "IVB (in.)",
                "hbreak_pctile": "HB (in.)",
                "CSW_pctile": "CSW (%)",
                "SwStr_pctile": "SwStr (%)",
                "LA_lte_0_pctile": "LA<=0%",
                "Z_Contact_pctile": "Z-Contact (%)",
                "Chase_pctile": "Chase (%)",
                "Ball_pct_pctile": "Ball (%)",
            }
            df = df.rename(columns=rename_map)
            df = df.sort_values(by="Pitch Grade Pctile", ascending=False)
            render_table(
                df,
                reverse_cols={"VAA", "Ball (%)", "Z-Contact (%)"},
                abs_cols=ABS_GRADIENT_COLS_PITCH_TYPES,
                label_cols=["Name", "Pitch Type", "Split", "split", "Split Type"],
            )
            download_button(df, "pitch_percentiles", "pitch_pct_download")


def pitch_comps():
    """Individual Pitches - Pitch Level Comps page (placeholder)"""
    st.title("Pitch Level Comparisons")

    st.info("Pitch-level comparison functionality coming soon!")
    st.write("This will allow you to find similar pitches based on shape and outcomes.")


def pitch_splits():
    """Individual Pitches - Splits page (placeholder)"""
    st.title("Individual Pitch Splits")

    if pitch_type_splits_df.empty:
        st.info("Missing pitch_types_splits.csv")
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
            split_df = pitch_type_splits_df[
                pitch_type_splits_df["split_type"] == split_type
            ].copy()
            if split_df.empty:
                available = sorted(
                    pitch_type_splits_df["split_type"].dropna().unique().tolist()
                )
                st.info(f"No data for {tab_name}. Available split types: {available}")
                continue

            left, right = st.columns([1, 3])
            with left:
                level = st.selectbox(
                    "Select Level",
                    ["All", "MLB", "Triple-A", "Low-A", "Low Minors"],
                    index=1,
                    key=f"pitch_splits_level_{idx}",
                )
                season = st.multiselect(
                    "Select Season",
                    season_options(split_df),
                    default=(
                        [season_options(split_df)[1]]
                        if len(season_options(split_df)) > 1
                        else ["All"]
                    ),
                    key=f"pitch_splits_season_{idx}",
                )
                min_pitches = st.number_input(
                    "Minimum # Pitches",
                    min_value=0,
                    max_value=1000,
                    value=50,
                    step=1,
                    key=f"pitch_splits_min_pitches_{idx}",
                )
                split_choice = st.multiselect(
                    "Select Split",
                    ["All"] + sorted(split_df["split"].dropna().unique().tolist()),
                    default=["All"],
                    key=f"pitch_splits_split_{idx}",
                )
                team = st.selectbox(
                    "Select Team",
                    team_options(split_df, "pitching_code"),
                    index=0,
                    key=f"pitch_splits_team_{idx}",
                )
                pitcher_options, pitcher_name_map = player_id_options(
                    split_df, "pitcher_mlbid", "name"
                )
                pitcher = st.multiselect(
                    "Select Pitcher",
                    pitcher_options,
                    default=["All"],
                    format_func=lambda v: (
                        "All"
                        if v == "All"
                        else f"{pitcher_name_map.get(v, 'Unknown')} ({int(v)})"
                    ),
                    key=f"pitch_splits_pitcher_{idx}",
                )
                pitch_group = st.multiselect(
                    "Select Pitch Group",
                    (
                        ["All"]
                        + sorted(split_df["pitch_group"].dropna().unique().tolist())
                        if "pitch_group" in split_df.columns
                        else ["All"]
                    ),
                    default=["All"],
                    key=f"pitch_splits_pitch_group_{idx}",
                )
                pitch_tag = st.multiselect(
                    "Select Pitch Type",
                    ["All"] + sorted(split_df["pitch_tag"].dropna().unique().tolist()),
                    default=["All"],
                    key=f"pitch_splits_pitch_tag_{idx}",
                )
            with right:
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
                df = filter_by_values(df, "split", split_choice)
                df = filter_by_team_token(df, "pitching_code", team)
                df = filter_by_values(df, "pitcher_mlbid", pitcher)
                if "pitch_group" in df.columns:
                    df = filter_by_values(df, "pitch_group", pitch_group)
                df = filter_by_values(df, "pitch_tag", pitch_tag)
                df = df[df["pitches"] >= min_pitches]
                df = df.assign(__season=df["season"], __level=df["level_id"])

                columns = [
                    "name",
                    "pitcher_mlbid",
                    "pitching_code",
                    "season",
                    "split",
                    "pitch_tag",
                    "pitches",
                    "pct",
                    "stuff",
                    "velo",
                    "max_velo",
                    "vaa",
                    "haa",
                    "vbreak",
                    "hbreak",
                    "SwStr",
                    "Z_Contact",
                    "Ball_pct",
                    "Zone",
                    "Chase",
                    "CSW",
                    "__season",
                    "__level",
                ]
                df = df[[col for col in columns if col in df.columns]].copy()
                if "stuff" in df.columns:
                    df["stuff"] = df["stuff"].round(0)
                rename_map = {
                    "name": "Name",
                    "pitcher_mlbid": "Player ID",
                    "pitching_code": "Team",
                    "season": "Season",
                    "split": "Split",
                    "pitch_tag": "Pitch Type",
                    "pitches": "#",
                    "pct": "Usage (%)",
                    "stuff": "Pitch Grade",
                    "velo": "Velo",
                    "max_velo": "Max Velo",
                    "vaa": "VAA",
                    "haa": "HAA",
                    "vbreak": "IVB (in.)",
                    "hbreak": "HB (in.)",
                    "CSW": "CSW (%)",
                    "SwStr": "SwStr (%)",
                    "Z_Contact": "Z-Contact (%)",
                    "Chase": "Chase (%)",
                    "Zone": "Zone (%)",
                    "Ball_pct": "Ball (%)",
                }
                df = df.rename(columns=rename_map)
                df = df.sort_values(by="Pitch Grade", ascending=False)
                stats_df = base_stats[
                    [col for col in columns if col in base_stats.columns]
                ].rename(columns=rename_map)
                render_table(
                    df,
                    reverse_cols={"Ball (%)", "Z-Contact (%)", "VAA"},
                    group_cols=["__season", "__level"],
                    stats_df=stats_df,
                    abs_cols=ABS_GRADIENT_COLS_PITCH_TYPES,
                    label_cols=["Name", "Pitch Type", "Split", "split", "Split Type"],
                )
                download_button(
                    df,
                    f"pitch_splits_{idx}",
                    f"pitch_splits_download_{idx}",
                )


# =============================================================================
# TEAMS PAGES
# =============================================================================


def team_hitting():
    """Team Hitting page"""
    st.title("Team Hitting")

    if team_damage.empty:
        st.info("Missing new_team_damage.csv")
    else:
        left, right = st.columns([1, 3])
        with left:
            level = st.selectbox(
                "Select Level",
                ["MLB", "Triple-A", "Low-A", "Low Minors"],
                index=0,
                key="team_hitting_level",
            )
            season = st.multiselect(
                "Select Season",
                season_options(team_damage),
                default=(
                    [season_options(team_damage)[1]]
                    if len(season_options(team_damage)) > 1
                    else ["All"]
                ),
                key="team_hitting_season",
            )
        with right:
            level_map = {
                "MLB": [1],
                "Triple-A": [11],
                "Low-A": [14],
                "Low Minors": [16],
            }
            base_stats = team_damage.copy()
            base_stats = base_stats.assign(
                __season=base_stats["season"], __level=base_stats["level_id"]
            )
            df = team_damage.copy()
            df = df[df["level_id"].isin(level_map[level])]
            df = filter_by_values(df, "season", season)
            df = df.assign(__season=df["season"], __level=df["level_id"])

            columns = [
                "hitting_code",
                "season",
                "PA",
                "bbe",
                "damage_rate",
                "EV90th",
                "pull_FB_pct",
                "LA_gte_20",
                "LA_lte_0",
                "SEAGER",
                "selection_skill",
                "hittable_pitches_taken",
                "chase",
                "z_con",
                "secondary_whiff_pct",
                "whiffs_vs_95",
                "contact_vs_avg",
                "__season",
                "__level",
            ]
            df = df[[col for col in columns if col in df.columns]].copy()
            rename_map = {
                "hitting_code": "Team",
                "season": "Season",
                "bbe": "BBE",
                "damage_rate": "Damage/BBE (%)",
                "EV90th": "90th Pctile EV",
                "pull_FB_pct": "Pulled FB (%)",
                "LA_gte_20": "LA>=20%",
                "LA_lte_0": "LA<=0%",
                "selection_skill": "Selectivity (%)",
                "hittable_pitches_taken": "Hittable Pitch Take (%)",
                "chase": "Chase (%)",
                "z_con": "Z-Contact (%)",
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
                include_team_label=True,
            )
            download_button(df, "team_hitting", "team_hitting_download")


def team_pitching():
    """Team Pitching page"""
    st.title("Team Pitching")

    if team_stuff.empty:
        st.info("Missing new_team_stuff.csv")
    else:
        left, right = st.columns([1, 3])
        with left:
            level = st.selectbox(
                "Select Level",
                ["MLB", "Triple-A", "Low-A", "Low Minors"],
                index=0,
                key="team_pitching_level",
            )
            season = st.multiselect(
                "Select Season",
                season_options(team_stuff),
                default=(
                    [season_options(team_stuff)[1]]
                    if len(season_options(team_stuff)) > 1
                    else ["All"]
                ),
                key="team_pitching_season",
            )
        with right:
            level_map = {
                "MLB": [1],
                "Triple-A": [11],
                "Low-A": [14],
                "Low Minors": [16],
            }
            base_stats = team_stuff.copy()
            base_stats = base_stats.assign(
                __season=base_stats["season"], __level=base_stats["level_id"]
            )
            df = team_stuff.copy()
            df = df[df["level_id"].isin(level_map[level])]
            df = filter_by_values(df, "season", season)
            df = df.assign(__season=df["season"], __level=df["level_id"])

            columns = [
                "pitching_code",
                "season",
                "IP",
                "stuff",
                "fastball_velo",
                "fastball_vaa",
                "FA_pct",
                "SwStr",
                "Ball_pct",
                "Z_Contact",
                "Chase",
                "CSW",
                "LA_lte_0",
                "__season",
                "__level",
            ]
            df = df[[col for col in columns if col in df.columns]].copy()
            rename_map = {
                "pitching_code": "Team",
                "season": "Season",
                "stuff": "Pitch Grade",
                "fastball_velo": "FA mph",
                "fastball_vaa": "FA VAA",
                "FA_pct": "FA Usage (%)",
                "SwStr": "SwStr (%)",
                "Ball_pct": "Ball (%)",
                "Z_Contact": "Z-Contact (%)",
                "Chase": "Chase (%)",
                "CSW": "CSW (%)",
                "LA_lte_0": "LA<=0%",
            }
            df = df.rename(columns=rename_map)
            df = df.sort_values(by="Pitch Grade", ascending=False)
            stats_df = base_stats[
                [col for col in columns if col in base_stats.columns]
            ].rename(columns=rename_map)
            render_table(
                df,
                reverse_cols={"Ball (%)", "FA VAA", "Z-Contact (%)"},
                group_cols=["__season", "__level"],
                stats_df=stats_df,
                include_team_label=True,
            )
            download_button(df, "team_pitching", "team_pitching_download")


# =============================================================================
# LEAGUE PAGES
# =============================================================================


def league_hitting():
    """League - Hitting Stats page"""
    st.title("League Averages - Hitting")

    if hitting_avg.empty:
        st.info("Missing new_hitting_lg_avg.csv")
    else:
        left, right = st.columns([1, 3])
        with left:
            season = st.multiselect(
                "Select Season",
                season_options(hitting_avg),
                default=(
                    [season_options(hitting_avg)[1]]
                    if len(season_options(hitting_avg)) > 1
                    else ["All"]
                ),
                key="lg_hit_season",
            )
        with right:
            df = hitting_avg.copy()
            df = filter_by_values(df, "season", season)
            df = df.assign(
                Level=df["level_id"].map(
                    {1: "MLB", 11: "Triple-A", 14: "Low-A", 16: "Low Minors"}
                )
            )
            base_stats = hitting_avg.copy()
            base_stats = base_stats.assign(
                __season=base_stats["season"], __level=base_stats["level_id"]
            )
            df = df.assign(__season=df["season"], __level=df["level_id"])
            columns = [
                "Level",
                "season",
                "PA",
                "bbe",
                "damage_rate",
                "EV90th",
                "pull_FB_pct",
                "LA_gte_20",
                "LA_lte_0",
                "SEAGER",
                "selection_skill",
                "hittable_pitches_taken",
                "chase",
                "z_con",
                "secondary_whiff_pct",
                "whiffs_vs_95",
                "contact_vs_avg",
                "__season",
                "__level",
            ]
            df = df[[col for col in columns if col in df.columns]].copy()
            rename_map = {
                "season": "Season",
                "bbe": "BBE",
                "damage_rate": "Damage/BBE (%)",
                "EV90th": "90th Pctile EV",
                "pull_FB_pct": "Pulled FB (%)",
                "LA_gte_20": "LA>=20%",
                "LA_lte_0": "LA<=0%",
                "selection_skill": "Selectivity (%)",
                "hittable_pitches_taken": "Hittable Pitch Take (%)",
                "chase": "Chase (%)",
                "z_con": "Z-Contact (%)",
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
                group_cols=["__season", "__level"],
                stats_df=stats_df,
            )
            download_button(df, "league_hitting", "league_hitting_download")


def league_pitching():
    """League - Pitching Stats page"""
    st.title("League Averages - Pitching")

    if pitching_avg.empty:
        st.info("Missing new_lg_stuff.csv")
    else:
        left, right = st.columns([1, 3])
        with left:
            level = st.selectbox(
                "Select Level",
                ["MLB", "Triple-A", "Low-A", "Low Minors"],
                index=0,
                key="lg_pitch_level",
            )
            season = st.multiselect(
                "Select Season",
                season_options(pitching_avg),
                default=(
                    [season_options(pitching_avg)[1]]
                    if len(season_options(pitching_avg)) > 1
                    else ["All"]
                ),
                key="lg_pitch_season",
            )
        with right:
            level_map = {
                "MLB": [1],
                "Triple-A": [11],
                "Low-A": [14],
                "Low Minors": [16],
            }
            base_stats = pitching_avg.copy()
            base_stats = base_stats.assign(
                __season=base_stats["season"], __level=base_stats["level_id"]
            )
            df = pitching_avg.copy()
            df = df[df["level_id"].isin(level_map[level])]
            df = filter_by_values(df, "season", season)
            df = df.assign(__season=df["season"], __level=df["level_id"])

            columns = [
                "season",
                "stuff",
                "stuff_z",
                "fastball_velo",
                "fastball_vaa",
                "FA_pct",
                "BB_rpm",
                "SwStr",
                "Ball_pct",
                "Z_Contact",
                "Chase",
                "CSW",
                "LA_lte_0",
                "__season",
                "__level",
            ]
            df = df[[col for col in columns if col in df.columns]].copy()
            # Round BB_rpm and stuff to integers
            if "BB_rpm" in df.columns:
                df["BB_rpm"] = df["BB_rpm"].round(0)
            if "stuff" in df.columns:
                df["stuff"] = df["stuff"].round(0)
            rename_map = {
                "season": "Season",
                "CSW": "CSW (%)",
                "Ball_pct": "Ball (%)",
                "SwStr": "SwStr (%)",
                "Z_Contact": "Z-Contact (%)",
                "Chase": "Chase (%)",
                "fastball_velo": "FA mph",
                "fastball_vaa": "FA VAA",
                "FA_pct": "FA Usage (%)",
                "BB_rpm": "BB Spin",
                "stuff": "Pitch Grade",
                "stuff_z": "Pitch Grade Z",
                "LA_lte_0": "LA<=0%",
            }
            df = df.rename(columns=rename_map)
            stats_df = base_stats[
                [col for col in columns if col in base_stats.columns]
            ].rename(columns=rename_map)
            render_table(
                df,
                group_cols=["__season", "__level"],
                stats_df=stats_df,
            )
            download_button(df, "league_pitching", "league_pitching_download")


def league_pitch_level():
    """League - Pitch Level Shapes and Outcomes page"""
    st.title("League Averages - Pitch Level Shapes and Outcomes")

    if league_pitch_types.empty:
        st.info("Missing league_pitch_types.csv")
        return
    if "throws" not in league_pitch_types.columns:
        st.info("league_pitch_types.csv is outdated. Please re-run data_aggregate.py.")
        return

    left, right = st.columns([1, 3])
    with left:
        season = st.multiselect(
            "Select Season",
            season_options(league_pitch_types),
            default=(
                [season_options(league_pitch_types)[1]]
                if len(season_options(league_pitch_types)) > 1
                else ["All"]
            ),
            key="lg_pitch_types_season",
        )
        throws = st.multiselect(
            "Select Throws",
            ["All"] + sorted(league_pitch_types["throws"].dropna().unique().tolist()),
            default=["All"],
            key="lg_pitch_types_throws",
        )
        pitch_tag = st.multiselect(
            "Select Pitch Type",
            ["All"]
            + sorted(league_pitch_types["pitch_tag"].dropna().unique().tolist()),
            default=["All"],
            key="lg_pitch_types_pitch_tag",
        )
    with right:
        base_stats = league_pitch_types.copy()
        base_stats = base_stats.assign(__season=base_stats["season"])
        df = league_pitch_types.copy()
        df = filter_by_values(df, "season", season)
        df = filter_by_values(df, "throws", throws)
        df = filter_by_values(df, "pitch_tag", pitch_tag)
        df = df.assign(__season=df["season"])

        columns = [
            "season",
            "throws",
            "pitch_tag",
            "pct",
            "velo",
            "vaa",
            "haa",
            "vbreak",
            "hbreak",
            "SwStr",
            "LA_lte_0",
            "Z_Contact",
            "Zone",
            "Ball_pct",
            "Chase",
            "CSW",
            "__season",
        ]
        df = df[[col for col in columns if col in df.columns]].copy()
        rename_map = {
            "season": "Season",
            "throws": "Throws",
            "pitch_tag": "Pitch Type",
            "pct": "Usage (%)",
            "velo": "Velo",
            "vaa": "VAA",
            "haa": "HAA",
            "vbreak": "IVB (in.)",
            "hbreak": "HB (in.)",
            "SwStr": "SwStr (%)",
            "LA_lte_0": "LA<=0%",
            "Z_Contact": "Z-Contact (%)",
            "Zone": "Zone (%)",
            "Ball_pct": "Ball (%)",
            "Chase": "Chase (%)",
            "CSW": "CSW (%)",
        }
        df = df.rename(columns=rename_map)
        stats_df = base_stats[
            [col for col in columns if col in base_stats.columns]
        ].rename(columns=rename_map)
        render_table(
            df,
            reverse_cols={"Ball (%)", "Z-Contact (%)", "VAA"},
            group_cols=["__season"],
            stats_df=stats_df,
        )
        download_button(df, "league_pitch_types", "league_pitch_types_download")


# =============================================================================
# PARKS PAGE
# =============================================================================


def park_data_page():
    """Parks - HR per Damage BBE page"""
    st.title("Park HR per Damage BBE")

    if park_data.empty:
        st.info("Missing park_data.csv (or park_data.parquet)")
        return

    left, right = st.columns([1, 3])
    with left:
        level = st.selectbox(
            "Select Level",
            ["All", "MLB", "Triple-A", "Low-A", "Low Minors"],
            index=1,
            key="park_level",
        )
        season = st.multiselect(
            "Select Season",
            season_options(park_data),
            default=(
                [season_options(park_data)[1]]
                if len(season_options(park_data)) > 1
                else ["All"]
            ),
            key="park_season",
        )
        stands = st.multiselect(
            "Select Batter Handedness",
            ["All"] + sorted(park_data["stands"].dropna().unique().tolist()),
            default=["All"],
            key="park_stands",
        )
        team = st.selectbox(
            "Select Home Team",
            ["All"] + sorted(park_data["home_team"].dropna().unique().tolist()),
            index=0,
            key="park_home_team",
        )
        park_pairs = (
            park_data[["park_mlbid", "home_team"]]
            .dropna()
            .drop_duplicates()
            .sort_values(by=["park_mlbid", "home_team"])
            .values.tolist()
        )
        park_options = ["All"] + [tuple(pair) for pair in park_pairs]
        park = st.selectbox(
            "Select Park",
            park_options,
            index=0,
            key="park_mlbid",
            format_func=lambda v: ("All" if v == "All" else f"{v[0]} - {v[1]}"),
        )
    with right:
        level_map = {
            "All": [1, 11, 14, 16],
            "MLB": [1],
            "Triple-A": [11],
            "Low-A": [14],
            "Low Minors": [16],
        }
        df = park_data.copy()
        df = df[df["level_id"].isin(level_map[level])]
        df = filter_by_values(df, "season", season)
        df = filter_by_values(df, "stands", stands)
        if team != "All":
            df = df[df["home_team"] == team]
        if park != "All":
            df = df[(df["park_mlbid"] == park[0]) & (df["home_team"] == park[1])]
        df = df.assign(
            Level=df["level_id"].map(
                {1: "MLB", 11: "Triple-A", 14: "Low-A", 16: "Low Minors"}
            ),
            __season=df["season"],
            __level=df["level_id"],
        )

        columns = [
            "park_mlbid",
            "home_team",
            "season",
            "stands",
            "Level",
            "damage_bbe",
            "HR_per_damage_BBE_pct",
            "XBH_per_damage_BBE_pct",
            "Hits_per_BBE_pct",
            "__season",
            "__level",
        ]
        df = df[[col for col in columns if col in df.columns]].copy()
        rename_map = {
            "park_mlbid": "Park ID",
            "home_team": "Home Team",
            "stands": "Batter Hand",
            "season": "Season",
            "damage_bbe": "Damage BBE",
            "HR_per_damage_BBE_pct": "HR per Damage BBE (%)",
            "XBH_per_damage_BBE_pct": "XBH per Damage BBE (%)",
            "Hits_per_BBE_pct": "Hits per BBE (%)",
        }
        df = df.rename(columns=rename_map)
        df = df.sort_values(by="HR per Damage BBE (%)", ascending=False)
        render_table(
            df,
            group_cols=["__season", "__level"],
            no_format_cols=DEFAULT_NO_FORMAT_COLS | {"Park ID", "Damage BBE"},
        )
        download_button(df, "park_data", "park_data_download")


# =============================================================================
# GLOSSARY PAGES
# =============================================================================


def glossary_hitting():
    """Glossary - Hitting page"""
    st.title("Glossary - Hitting")

    st.markdown(
        """
### Hitting Metrics Glossary

**Damage/BBE (%)**: Percentage of batted ball events that result in "damage" (extra-base hits or hard-hit balls likely to result in positive outcomes).

**90th Pctile EV**: The 90th percentile exit velocity for a player's batted balls.

**Pulled FB (%)**: Percentage of fly balls that are pulled to the pull side.

**LA>=20%**: Percentage of batted balls with launch angle of 20 degrees or higher (fly balls).

**LA<=0%**: Percentage of batted balls with launch angle of 0 degrees or lower (ground balls).

**SEAGER**: A composite metric measuring overall hitting quality and approach.

**Selectivity (%)**: Measure of a hitter's ability to swing at strikes and take balls.

**Hittable Pitch Take (%)**: Percentage of hittable pitches that the batter takes (does not swing at).

**Chase (%)**: Percentage of pitches outside the zone that the batter swings at.

**Z-Contact (%)**: Contact rate on pitches in the strike zone.

**Whiff vs. Secondaries (%)**: Whiff rate against secondary pitches (breaking balls, offspeed).

**Whiff vs. 95+ (%)**: Whiff rate against fastballs 95 mph or higher.

**Contact Over Expected (%)**: Contact rate compared to expected contact rate based on pitch characteristics. Only applied to hitter swings.
"""
    )


def glossary_pitching():
    """Glossary - Pitching page"""
    st.title("Glossary - Pitching")

    st.markdown(
        """
### Pitching Metrics Glossary

**Pitch Grade**: Overall pitch quality metric. Higher is better. Max is 80, min is 20. League median is typically within a few points of 50. Applied within pitch types.

**FA mph**: Average fastball velocity.

**Max FA mph**: Maximum fastball velocity.

**FA VAA**: Fastball vertical approach angle.

**FA Usage (%)**: Percentage of pitches that are fastballs.

**BB Spin**: Avg spin rate (RPM) of a pitcher's breaking balls.

**SwStr (%)**: Swinging strike percentage.

**Ball (%)**: Percentage of pitches resulting in balls.

**Z-Contact (%)**: Contact rate on pitches in the strike zone.

**Chase (%)**: Percentage of pitches outside the zone that induce swings.

**CSW (%)**: Called strikes plus whiffs percentage.

**LA<=0%**: Percentage of batted balls with launch angle of 0 degrees or lower (ground balls).

**Vertical Release (ft.)**: Vertical release point in feet.

**Horizontal Release (ft.)**: Horizontal release point in feet.

**Extension (ft.)**: Release point extension toward home plate in feet.

**VAA**: Vertical approach angle (for individual pitches).

**HAA**: Horizontal approach angle (for individual pitches).

**IVB (in.)**: Induced vertical break in inches.

**HB (in.)**: Horizontal break in inches.

**Zone (%)**: Percentage of pitches thrown in the strike zone.
"""
    )


# =============================================================================
# NAVIGATION SETUP
# =============================================================================

# Step 1: Check if user is logged in
try:
    is_logged_in = st.user.is_logged_in
except AttributeError:
    is_logged_in = False

if not is_logged_in:
    st.subheader("🔐 Login Required")
    st.markdown(
        """
        Please log in to access the premium features of this app.
        """
    )
    if st.button("Log in with Google", type="primary"):
        st.login()
    st.stop()

# Step 2: User is logged in, now check subscription
st.markdown(f"Welcome back, **{st.user.name}**! 👋")
st.markdown("---")

with st.expander("Manage subscription"):
    subscription_result = add_auth(
        required=False,
        show_redirect_button=False,
        subscription_button_text="Subscribe to Access Premium Features",
        button_color="#FF4B4B",
    )
    is_subscribed = _resolve_subscription_status(subscription_result)
    if not is_subscribed:
        st.markdown("Choose a plan:")
        plan_col_annual, plan_col_monthly = st.columns(2)
        with plan_col_annual:
            st.link_button("Annual ($40.00)", ANNUAL_PAYMENT_LINK)
        with plan_col_monthly:
            st.link_button("Monthly ($5.00)", MONTHLY_PAYMENT_LINK)
        st.caption("Use the same email as your Google login when subscribing.")
    st.write("Cancel, pause, or update your subscription via Stripe.")
    billing_email = _get_user_email()
    if not billing_email:
        st.info("Login with an email address to manage your Stripe subscription.")
    else:
        if "billing_portal_url" not in st.session_state:
            st.session_state.billing_portal_url = None
        if st.button("Open Stripe billing portal"):
            portal_url = _create_billing_portal_url(billing_email)
            if not portal_url:
                st.error(
                    "We couldn't open the billing portal. Make sure the subscription "
                    "was purchased with this email and set `billing_portal_return_url` "
                    "in `.streamlit/secrets.toml`."
                )
            else:
                st.session_state.billing_portal_url = portal_url
        if st.session_state.billing_portal_url:
            st.link_button(
                "Continue to billing portal", st.session_state.billing_portal_url
            )
if is_subscribed:
    st.success("You have premium access! Enjoy all features.")
else:
    st.info(
        f"Preview mode enabled. Tables are limited to the first {PREVIEW_ROWS} rows."
    )
st.markdown("---")

# ── Session timeout ───────────────────────────────────────────────────────────
import time as _time

SESSION_TIMEOUT_MINUTES = 30
if "last_activity" not in st.session_state:
    st.session_state.last_activity = _time.time()
else:
    idle_minutes = (_time.time() - st.session_state.last_activity) / 60
    if idle_minutes > SESSION_TIMEOUT_MINUTES:
        st.warning(
            "⏱️ Session timed out after 30 minutes of inactivity. Please refresh."
        )
        st.stop()
st.session_state.last_activity = _time.time()
# ─────────────────────────────────────────────────────────────────────────────
# Define page navigation with hierarchical groups
pages = {
    "Home": [
        st.Page(home_page, title="Welcome", icon="🏠"),
    ],
    "Hitters": [
        st.Page(hitter_individual_stats, title="Individual Stats", icon="⚾"),
        st.Page(hitter_percentiles, title="Percentiles", icon="📊"),
        st.Page(hitter_comps, title="Hitter Comps", icon="🔍"),
        st.Page(hitter_mlb_equivalencies, title="MLB Equivalencies", icon="🔁"),
        st.Page(hitter_ar, title="Auto Regressed (AR)", icon="📈"),
        st.Page(hitter_splits, title="Splits", icon="📋"),
    ],
    "Pitchers": [
        st.Page(pitcher_individual_stats, title="Individual Stats", icon="⚾"),
        st.Page(pitcher_percentiles, title="Percentiles", icon="📊"),
        st.Page(pitcher_comps, title="Pitcher Comps", icon="🔍"),
        st.Page(pitcher_mlb_equivalencies, title="MLB Equivalencies", icon="🔁"),
        st.Page(pitcher_ar, title="Auto Regressed (AR)", icon="📈"),
        st.Page(pitcher_splits, title="Splits", icon="📋"),
    ],
    "Individual Pitches": [
        st.Page(pitch_shapes_outcomes, title="Shapes and Outcomes", icon="🎯"),
        st.Page(pitch_ar, title="Auto Regressed (AR)", icon="📈"),
        st.Page(pitch_percentiles, title="Percentiles", icon="📊"),
        st.Page(pitch_comps, title="Pitch Level Comps", icon="🔍"),
        st.Page(pitch_splits, title="Splits", icon="📋"),
    ],
    "Teams": [
        st.Page(team_hitting, title="Team Hitting", icon="🏆"),
        st.Page(team_pitching, title="Team Pitching", icon="🏆"),
    ],
    "League": [
        st.Page(league_hitting, title="Hitting Stats", icon="🌐"),
        st.Page(league_pitching, title="Pitching Stats", icon="🌐"),
        st.Page(league_pitch_level, title="Pitch Level Shapes", icon="🌐"),
    ],
    "Parks": [
        st.Page(park_data_page, title="Park HR per Damage BBE", icon="🏟️"),
    ],
    "Glossary": [
        st.Page(glossary_hitting, title="Hitting Glossary", icon="📖"),
        st.Page(glossary_pitching, title="Pitching Glossary", icon="📖"),
    ],
}

# Create and run navigation
pg = st.navigation(pages)
pg.run()

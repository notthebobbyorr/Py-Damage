from __future__ import annotations

import sys
import time as _time
from pathlib import Path

import streamlit as st
from st_paywall import add_auth

from app.auth import (
    _create_billing_portal_url,
    _get_user_email,
    _is_subscription_exempt_user,
    _is_user_subscribed,
)
from app.config import ANNUAL_PAYMENT_LINK, MONTHLY_PAYMENT_LINK, PREVIEW_ROWS
from app.pages.glossary import glossary_hitting, glossary_pitching
from app.pages.hitters import (
    hitter_ar,
    hitter_comps,
    hitter_gamelogs_page,
    hitter_individual_stats,
    hitter_mlb_equivalencies,
    hitter_percentiles,
    hitter_splits,
)
from app.pages.home import home_page, home_timeline
from app.pages.league import league_hitting, league_pitch_level, league_pitching
from app.pages.parks import park_data_page
from app.pages.pitchers import (
    pitcher_ar,
    pitcher_comps,
    pitcher_gamelogs_page,
    pitcher_individual_stats,
    pitcher_mlb_equivalencies,
    pitcher_percentiles,
    pitcher_splits,
)
from app.pages.pitches import (
    pitch_ar,
    pitch_comps,
    pitch_percentiles,
    pitch_shapes_outcomes,
    pitch_splits,
    pitch_type_gamelogs_page,
)
from app.pages.teams import (
    team_hitting,
    team_hitting_gamelogs,
    team_pitching,
    team_pitching_gamelogs,
)
import app.viz as _viz


# ---------------------------------------------------------------------------
# CLI entry-point helpers
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Page config & global CSS
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Authentication — step 1: login gate
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Authentication — step 2: subscription check
# ---------------------------------------------------------------------------

st.markdown(f"Welcome back, **{st.user.name}**! 👋")
st.markdown("---")

with st.expander("Manage subscription"):
    subscription_result = add_auth(
        required=False,
        show_redirect_button=False,
        subscription_button_text="Subscribe to Access Premium Features",
        button_color="#FF4B4B",
    )
    is_subscribed = _is_user_subscribed(subscription_result)
    if not is_subscribed:
        st.markdown("Choose a plan:")
        plan_col_annual, plan_col_monthly = st.columns(2)
        with plan_col_annual:
            st.link_button("Annual ($40.00)", ANNUAL_PAYMENT_LINK)
        with plan_col_monthly:
            st.link_button("Monthly ($5.00)", MONTHLY_PAYMENT_LINK)
        st.caption("Use the same email as your Google login when subscribing.")
    elif _is_subscription_exempt_user():
        st.caption("Subscription exception active for this account.")
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

# ---------------------------------------------------------------------------
# Session timeout
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Page navigation
# ---------------------------------------------------------------------------

pages = {
    "Home": [
        st.Page(home_page, title="Welcome", icon="🏠"),
        st.Page(home_timeline, title="What's New", icon="🆕"),
    ],
    "Hitters": [
        st.Page(hitter_individual_stats, title="Individual Stats", icon="⚾"),
        st.Page(hitter_percentiles, title="Percentiles", icon="📊"),
        st.Page(hitter_comps, title="Hitter Comps", icon="🔍"),
        st.Page(hitter_mlb_equivalencies, title="MLB Equivalencies", icon="🔁"),
        st.Page(hitter_ar, title="Auto Regressed (AR)", icon="📈"),
        st.Page(hitter_splits, title="Splits", icon="📋"),
        st.Page(hitter_gamelogs_page, title="Gamelogs", icon="📅"),
    ],
    "Pitchers": [
        st.Page(pitcher_individual_stats, title="Individual Stats", icon="⚾"),
        st.Page(pitcher_percentiles, title="Percentiles", icon="📊"),
        st.Page(pitcher_comps, title="Pitcher Comps", icon="🔍"),
        st.Page(pitcher_mlb_equivalencies, title="MLB Equivalencies", icon="🔁"),
        st.Page(pitcher_ar, title="Auto Regressed (AR)", icon="📈"),
        st.Page(pitcher_splits, title="Splits", icon="📋"),
        st.Page(pitcher_gamelogs_page, title="Gamelogs", icon="📅"),
    ],
    "Individual Pitches": [
        st.Page(pitch_shapes_outcomes, title="Shapes and Outcomes", icon="🎯"),
        st.Page(pitch_percentiles, title="Percentiles", icon="📊"),
        st.Page(pitch_comps, title="Pitch Level Comps", icon="🔍"),
        st.Page(pitch_ar, title="Auto Regressed (AR)", icon="📈"),
        st.Page(pitch_splits, title="Splits", icon="📋"),
        st.Page(pitch_type_gamelogs_page, title="Gamelogs", icon="📅"),
    ],
    "Teams": [
        st.Page(team_hitting, title="Team Hitting", icon="🏆"),
        st.Page(team_pitching, title="Team Pitching", icon="🏆"),
        st.Page(team_hitting_gamelogs, title="Hitting Gamelogs", icon="📅"),
        st.Page(team_pitching_gamelogs, title="Pitching Gamelogs", icon="📅"),
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

pg = st.navigation(pages)
_viz._TABLE_COUNTER = 0
pg.run()

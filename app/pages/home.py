from __future__ import annotations

import pandas as pd
import streamlit as st


FEATURE_TIMELINE: list[dict[str, str]] = [
    {
        "date": "2026-05-15",
        "title": "Percentile Modes — Customizable vs. Constant",
        "page": "Hitters / Pitchers / Individual Pitches → Percentiles",
        "description": (
            "Each percentile page now has a **Percentile Mode** toggle at the top of the filter panel. \n\n"
            "**Customizable** (default) keeps the existing behavior — percentiles recompute from the population after applying filters. \n\n "
            "**Constant** uses a fixed, stable percentile basis drawn from the Regular Season, 150+ PA for Hitters / 150+ TBF for Pitchers / 150+ pitches for Individual Pitches per season and level. "
            "Tip: Use Constant when viewing a single player across multiple years — otherwise the dynamic recompute would rank the player only against themselves and return 100th percentile in every category. "
        ),
    },
    {
        "date": "2026-05-12",
        "title": "Team Splits, New Stats, & Pitch-Level Comps",
        "page": "Teams → Splits / Hitters → Individual Stats / Individual Pitches → Shapes and Outcomes / Individual Pitches → Pitch Level Comps",
        "description": (
            " - Added team-level splits (vs. L/R, Home/Away, 1st/2nd Half, Monthly) on the Team Hitting and Team Pitching pages. \n\n"
          " - Added swing metrics (VBA, or swing tilt, Avg Swing Speed, Avg Swing Length, Fast Swing (75+ mph) %, Intercept X (in. from batter's body horizontally), Intercept Y (in. from batter's body towards the pitcher) to the Hitter stats page. \n\n"
          " - Added Inferred Arm Angle to the Pitcher stats page. \n\n"
          " - Added Vertical and Horizontal Release Angles to the Individual Pitches shapes and outcomes page. \n\n"
          " - Added Pitch Level Comps for finding similar pitch-seasons by velo, shapes, release traits, or results. Every column from the Shapes and Outcomes page is available to form the comparison, and pitches are compared within broad pitch group buckets (fastballs to other fastballs, breaking balls to other breaking balls, etc.)."
        ),
    },
]


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


def home_timeline():
    """What's New — chronological log of feature/page additions."""
    st.title("What's New")
    st.caption(
        "A running log of new features, pages, and notable changes — newest first. "
        "Use this to catch up on anything that's been added since your last visit."
    )
    st.markdown("---")

    if not FEATURE_TIMELINE:
        st.info("No entries yet — check back soon.")
        return

    sorted_entries = sorted(
        FEATURE_TIMELINE, key=lambda e: e.get("date", ""), reverse=True
    )
    for entry in sorted_entries:
        date = entry.get("date", "")
        title = entry.get("title", "")
        page = entry.get("page", "")
        description = entry.get("description", "")
        try:
            date_dt = pd.to_datetime(date)
            date_label = f"{date_dt.strftime('%B')} {date_dt.day}, {date_dt.year}"
        except Exception:
            date_label = date or ""
        st.markdown(f"### {date_label} — {title}")
        if page:
            st.caption(f"📍 {page}")
        st.markdown(description)
        st.markdown("---")

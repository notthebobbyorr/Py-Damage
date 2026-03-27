from __future__ import annotations

import pandas as pd
import streamlit as st


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

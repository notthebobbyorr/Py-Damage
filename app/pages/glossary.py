from __future__ import annotations

import streamlit as st


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

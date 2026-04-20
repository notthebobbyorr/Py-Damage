from __future__ import annotations

import streamlit as st


def glossary_hitting():
    """Glossary - Hitting page"""
    st.title("Glossary - Hitting")

    st.markdown("""
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
                
**pSwing (%)**: Percentage of pitches the hitter would be expected to swing at based on count, pitch location, and pitch traits.
                
**SBO**: Stolen Base Opportunities. Number of times a player was on base with the base ahead of him unoccupied.
                
**SB**: Stolen Bases.

**Takeoff (%)**: Percentage of SBO where the player attempted to steal the next base.                                
""")


def glossary_pitching():
    """Glossary - Pitching page"""
    st.title("Glossary - Pitching")

    st.markdown("""
### Pitching Metrics Glossary

**Pitch Grade**: Overall pitch quality metric. Higher is better. Max is 80, min is 20. League median is typically within a few points of 50. Applied within pitch types.

**Execution Grade**: Pitch execution metric. Factors in location and tunneling with the rest of the pitcher's arsenal. Higher is better. Max is 80, min is 20. League median is typically within a few points of 50.

**FA mph**: Average fastball velocity.

**Max FA mph**: Maximum fastball velocity.

**FA VAA**: Vertical approach angle of that pitcher's most-used fastball. Lower values indicate a flatter fastball. A good rule of thumb is that a 4 or lower is an elite "rising" fastball. League average is typically around 5.

**FA Usage (%)**: Percentage of pitches that are fastballs (4-seam fastballs, sinkers, or hard cutters).

**BB Spin**: Avg spin rate (RPM) of a pitcher's breaking balls.

**SwStr (%)**: Swinging strike percentage.
                
**pSwStr (%)**: Expected swinging strike percentage based on pitch characteristics.

**pSwing (%)**: Percentage of pitches the pitcher would be expected to force a swing on based on count, pitch location, and pitch traits.
                
**Damage/BBE (%)**: Percentage of batted ball events that result in "damage" (likely extra-base hits or hard-hit balls) against that pitcher.
                
**pDamage/BBE (%)**: Expected percentage of batted ball events that would result in "damage" based on pitch characteristics.
                
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
                
**SBO**: Stolen Base Opportunities. Number of times a player was on base with the base ahead of him unoccupied.
                
**SB**: Stolen Bases.

**Takeoff Against (%)**: Percentage of SBO where a player attempted to steal the next base against the pitcher.        
""")

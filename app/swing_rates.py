"""Observed swing rates from counts, with equivalent rate identities as fallback."""
import numpy as np
import pandas as pd


def add_swing_rates(frame):
    if frame.empty or not any(c in frame for c in ['rate_whiffs', 'whiff_rate_num', 'SwStr', 'SwStr_num', 'whiffs', 'whiff_pct']):
        return frame
    frame = frame.copy()

    def first(names):
        value = pd.Series(np.nan, index=frame.index, dtype=float)
        for name in names:
            if name in frame:
                value = value.fillna(pd.to_numeric(frame[name], errors='coerce'))
        return value

    whiffs = first(['rate_whiffs', 'whiff_rate_num', 'SwStr_num', 'whiffs'])
    swings = first(['rate_swings', 'whiff_rate_den', 'Swing_pct_num', 'swings'])
    pitches = first(['rate_pitches', 'pitches', 'Swing_pct_den', 'SwStr_den'])
    swing = (100 * swings / pitches.where(pitches > 0)).fillna(first(['Swing_pct']))
    swstr = (100 * whiffs / pitches.where(pitches > 0)).fillna(first(['SwStr']))
    whiff = (100 * whiffs / swings.where(swings > 0)).fillna(first(['whiff_pct']))
    whiff = whiff.fillna(100 * swstr / swing.where(swing > 0))
    swstr = swstr.fillna(whiff * swing / 100)
    frame['Swing_pct'], frame['SwStr'], frame['whiff_pct'] = swing, swstr, whiff
    return frame

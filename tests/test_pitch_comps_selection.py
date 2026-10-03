import unittest

from streamlit.testing.v1 import AppTest


SCRIPT = """
import pandas as pd
import streamlit as st
from app.pages import pitches
rows = []
for level, game_type, velo in [(1, 'Regular Season', 90), (11, 'Regular Season', 95), (11, 'Postseason', 97)]:
    rows.append(dict(pitcher_mlbid=1, name='Target', pitching_code='NYY', season=2026,
                     level_id=level, game_type_group=game_type, pitches=50,
                     pitch_tag='FF', pitch_group='FA', velo=velo))
for player_id, level, velo in [(2, 1, 94), (3, 1, 96), (4, 11, 95)]:
    rows.append(dict(pitcher_mlbid=player_id, name=f'Comp {player_id}', pitching_code='BOS',
                     season=2026, level_id=level, game_type_group='Regular Season',
                     pitches=200, pitch_tag='FF', pitch_group='FA', velo=velo))
pitches.pitch_types = pd.DataFrame(rows)
captured = []
pitches.render_table = lambda frame, **kwargs: captured.append(frame)
pitches.download_button = lambda *args, **kwargs: None
pitches.pitch_comps()
st.session_state['captured'] = captured
"""


class PitchCompsSelectionTests(unittest.TestCase):
    def test_level_and_game_type_select_exact_target_keep_mlb_pool(self):
        app = AppTest.from_string(SCRIPT, default_timeout=120).run()
        self.assertFalse(app.exception)
        app.selectbox(key='pitch_comps_target_level').select('Triple-A').run()
        app.selectbox(key='pitch_comps_player').select(1).run()
        target, comps = app.session_state['captured']
        self.assertEqual(target['__level'].tolist(), [11])
        self.assertTrue((comps['__level'] == 1).all())
        app.selectbox(key='pitch_comps_game_type').select('Postseason').run()
        self.assertFalse(app.exception)
        target, comps = app.session_state['captured']
        self.assertEqual(len(target), 1)
        self.assertIn(97, target.iloc[0].values)
        self.assertTrue((comps['__level'] == 1).all())
        app.selectbox(key='pitch_comps_target_level').select('Low-A').run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['captured'], [])
        self.assertIn('No eligible Low-A', app.info[0].value)

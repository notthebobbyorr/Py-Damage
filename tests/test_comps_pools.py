import unittest
from streamlit.testing.v1 import AppTest


class ComparisonPoolTests(unittest.TestCase):
    def test_reverse_searches_and_workload_do_not_change_target(self):
        for kind, singular, id_col, name_col, feature in [
            ("hitters", "hitter", "batter_mlbid", "hitter_name", "damage_rate_reg"),
            ("pitchers", "pitcher", "pitcher_mlbid", "name", "fastball_velo_reg"),
            ("pitches", "pitch", "pitcher_mlbid", "name", "velo"),
        ]:
            with self.subTest(kind=kind):
                script = f"""
import pandas as pd
import streamlit as st
from app.pages import {kind} as page
rows = []
for player, level, workload in [(1,1,50),(1,11,50),(2,1,300),(3,11,300),(4,11,80)]:
    rows.append(dict({id_col}=player, {name_col}=f'Player {{player}}',
        season=2026, level_id=level, game_type_group='Regular Season',
        hitting_code='NYM', pitching_code='NYM', PA=workload, TBF=workload,
        IP=workload/3, GS=10, pitches=workload, bbe=workload,
        pitch_tag='FF', pitch_group='FA', {feature}=90+player,
        {feature}_mlb_eq=90+player))
# Postseason copies must never add extra target or candidate rows.
rows += [dict(row, game_type_group='Postseason') for row in rows]
frame = pd.DataFrame(rows)
# Nullable source metrics must reach NumPy as numeric arrays, not objects.
frame['{feature}'] = frame['{feature}'].astype('Float64')
frame['{feature}_mlb_eq'] = frame['{feature}_mlb_eq'].astype('Float64')
hand_col = 'batter_hand' if '{kind}' == 'hitters' else 'pitcher_hand'
frame[hand_col] = frame['{id_col}'].map({{1:'R',2:None,3:'L',4:'S' if '{kind}' == 'hitters' else 'R'}})
if '{kind}' == 'pitches':
    page.pitch_types = frame
else:
    setattr(page, '{kind}_reg_df', frame)
    setattr(page, '{kind}_mlb_eq_df', frame)
captured = []
page.render_table = lambda frame, **kwargs: captured.append(frame)
page.download_button = lambda *args, **kwargs: None
page.{singular}_comps()
st.session_state['captured'] = captured
"""
                app = AppTest.from_string(script, default_timeout=120).run()
                self.assertFalse(app.exception)
                if kind != "pitches":
                    app.toggle(key=f"{singular}_comps_use_mlb_eq").set_value(True).run()
                app.selectbox(key=f"{singular}_comps_player").select(1).run()
                app.selectbox(key=f"{singular}_comps_result_pool").select("Minor leagues").run()
                self.assertFalse(app.exception)
                target, comps = app.session_state['captured']
                self.assertEqual(comps['Name'].tolist(), ['Player 3'])
                app.selectbox(key=f"{singular}_comps_result_pool").select("MLB + minor leagues").run()
                mixed = app.session_state['captured'][1]
                self.assertEqual(dict(zip(mixed['Name'], mixed['Level'])),
                                 {'Player 2': 'MLB', 'Player 3': 'Triple-A'})
                self.assertEqual(mixed.columns.get_loc('Level'), mixed.columns.get_loc('Season') + 1)
                app.selectbox(key=f"{singular}_comps_result_pool").select("Minor leagues").run()
                metric = 'PA' if kind == 'hitters' else 'TBF' if kind == 'pitchers' else 'pitches'
                app.number_input(key=f"{singular}_comps_{metric}_minimum").set_value(60.0).run()
                new_target, comps = app.session_state['captured']
                self.assertTrue(target.equals(new_target))
                self.assertEqual(set(comps['Name']), {'Player 3','Player 4'})
                hand_col = 'batter_hand' if kind == 'hitters' else 'pitcher_hand'
                hand_key = f"{singular}_comps_{hand_col}"
                app.multiselect(key=hand_key).set_value(['L']).run()
                self.assertFalse(app.exception)
                hand_target, hand_comps = app.session_state['captured']
                self.assertTrue(target.equals(hand_target))
                self.assertEqual(hand_comps['Name'].tolist(), ['Player 3'])
                app.multiselect(key=hand_key).set_value(['S' if kind == 'hitters' else 'R']).run()
                self.assertEqual(app.session_state['captured'][1]['Name'].tolist(), ['Player 4'])
                app.multiselect(key=hand_key).set_value(['All']).run()

                app.selectbox(key=f"{singular}_comps_target_level").select('Triple-A' if kind == 'pitches' else 11).run()
                self.assertFalse(app.exception)
                self.assertEqual(set(app.session_state['captured'][1]['Name']), {'Player 3','Player 4'})
                app.checkbox(key=f"{singular}_comps_workload_max").check().run()
                app.number_input(key=f"{singular}_comps_{metric}_maximum").set_value(100.0).run()
                self.assertEqual(app.session_state['captured'][1]['Name'].tolist(), ['Player 4'])
                app.number_input(key=f"{singular}_comps_{metric}_minimum").set_value(400.0).run()
                self.assertFalse(app.exception)
                self.assertEqual(app.session_state['captured'], [])

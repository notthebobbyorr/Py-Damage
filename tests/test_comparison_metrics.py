import ast
from pathlib import Path
import unittest

import polars as pl
from streamlit.testing.v1 import AppTest

from app.comparison_metrics import MAIN_METRICS, comparison_metric, BAT_METRICS
from app.comps import prepare_comparison_metrics
from pipeline.comparison_rates import comparison_rate_counts
from pipeline.apply_regression_from_agg import apply_rate_from_agg
from pipeline.blend_constants import build_blended_rows, compute_current_mu


class ComparisonMetricsTests(unittest.TestCase):
    def test_daily_blend_recognizes_new_regular_season_rate_definitions(self):
        import yaml
        config=yaml.safe_load(Path('config/stability_config.yml').read_text())
        config['stats']={role:[s for s in config['stats'][role] if s['name']=='whiff_pct']
                         for role in ['hitters','pitchers']}
        rows=[]
        for player in range(20):
            for game_type in ['R','S']:
                for pitch in range(100):
                    rows.append(dict(batter_mlbid=player,pitcher_mlbid=player,
                        season=2026,level_id=1,game_type=game_type,swing=True,
                        whiff=(pitch < [10,20,30,40][player%4]) if game_type=='R' else True))
        current=compute_current_mu(pl.DataFrame(rows),config,20)
        self.assertEqual(current.height,2)
        self.assertEqual(current['n_players_current'].to_list(),[20,20])
        for value in current['mu_current']:
            self.assertAlmostEqual(value,.25)

    def test_observed_exceptions_never_use_modeled_aliases(self):
        import pandas as pd
        from app.utils import _hitter_display_map, _pitcher_display_map, _pitch_display_map
        for role, label_map in [('hitter', _hitter_display_map()),
                ('pitcher', _pitcher_display_map()), ('pitch', _pitch_display_map())]:
            for col in MAIN_METRICS[role]:
                self.assertIn(col, label_map, (role,col))
        for col in BAT_METRICS:
            self.assertEqual(comparison_metric('hitter',col+'_reg_mlb_eq',True),col)
        self.assertEqual(comparison_metric('pitcher','FA_pct_reg_mlb_eq',True),'FA_pct')
        frame = pd.DataFrame({'bat_speed': [70.], 'bat_speed_reg': [60.],
                              'miss_distance': [2.], 'whiff_pct': [10.]})
        result = prepare_comparison_metrics(frame,'hitter',True)
        self.assertNotIn('miss_distance_mlb_eq',result)
        self.assertNotIn('whiff_pct_reg',result)
        self.assertEqual(result['bat_speed'].iloc[0],70.)

    def test_main_page_metrics_are_covered_without_counts(self):
        excluded = {'hitter_name', 'batter_mlbid', 'hitting_code', 'name', 'pitcher_mlbid',
            'pitching_code', 'season', 'pitch_tag', 'pitches', 'bbe', 'HR', 'SBO', 'SB',
            '__season', '__level'}
        for role, page, function in [('hitter','hitters','hitter_individual_stats'),
                ('pitcher','pitchers','pitcher_individual_stats'), ('pitch','pitches','pitch_shapes_outcomes')]:
            tree = ast.parse(Path(f'app/pages/{page}.py').read_text(encoding='utf-8'))
            node = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function)
            columns = next(ast.literal_eval(n.value) for n in ast.walk(node)
                if isinstance(n,ast.Assign) and isinstance(n.value,ast.List)
                and any(isinstance(t,ast.Name) and t.id=='columns' for t in n.targets))
            covered = {c.removesuffix('_reg') for c in MAIN_METRICS[role]}
            self.assertFalse(set(columns) - excluded - covered, role)
            self.assertFalse(set(MAIN_METRICS[role]) & excluded, role)

    def test_modes_expose_modeled_skills_and_observed_exceptions(self):
        for role in MAIN_METRICS:
            for equivalent in ([False, True] if role != 'pitch' else [False]):
                script = f'''
import pandas as pd
import streamlit as st
from app.comps import comparison_columns
from app.comparison_metrics import MAIN_METRICS,comparison_metric
role={role!r}
equivalent={equivalent!r}
columns=[comparison_metric(role,c,equivalent) for c in MAIN_METRICS[role]]
frame=pd.DataFrame({{c:[1.,2.] for c in columns+['bbe','HR','PA','SBO','pitches','whiff_pct']}})
selected,labels=comparison_columns(role,frame,{{}},[],'metrics',equivalent,available=['bbe','HR','PA','SBO','pitches'])
st.session_state['selected']=selected
st.session_state['expected_options']=[labels.get(c,c) for c in columns]
'''
                app = AppTest.from_string(script).run()
                self.assertFalse(app.exception)
                options = app.multiselect(key='metrics').options
                expected = [comparison_metric(role,c,equivalent) for c in MAIN_METRICS[role]]
                self.assertEqual(set(options), set(app.session_state['expected_options']))
                app.multiselect(key='metrics').set_value(expected).run()
                self.assertFalse(app.exception)
                self.assertEqual(set(app.session_state['selected']), set(expected))
                if role != 'pitch':
                    self.assertNotIn('whiff_pct', options)
                    self.assertIn('miss_distance', options)

    def test_rates_use_stored_priors_without_refitting_or_raw_fallback(self):
        frame = pl.DataFrame(dict(pitcher_mlbid=[1,2], season=[2026]*2,
            level_id=[1,2], game_type_group=['Regular Season']*2,
            rate_whiffs=[10.,10.], rate_swings=[100.,100.],
            Swing_pct_num=[100.,100.], Swing_pct_den=[200.,200.],
            Zone_num=[90.,90.], Zone_den=[200.,200.]))
        constants=pl.DataFrame(dict(dataset=['pitchers'],stat=['whiff_pct'],
            season=[2026],level_id=[1],mu=[.25],K=[40.]))
        keys=['pitcher_mlbid','season','level_id','game_type_group']
        result=apply_rate_from_agg(comparison_rate_counts(frame,'pitchers'),constants,
            'pitchers','whiff_pct',keys,require_prior=True)
        self.assertAlmostEqual(result['whiff_pct_reg'][0],100*(10+40*.25)/140)
        self.assertIsNone(result['whiff_pct_reg'][1])
        self.assertEqual(result['whiff_pct_raw'][1],10.)

    def test_new_and_existing_whiff_measures_share_seasonal_blending(self):
        prior=pl.DataFrame(dict(dataset=['hitters']*2,
            stat=['whiff_pct','secondary_whiff_pct'],level_id=[1]*2,
            season=[2025.]*2,kind=['rate']*2,mu=[.2]*2,K=[40.]*2,
            n_players=[100]*2,n_50=[40.]*2,n_70=[40*7/3]*2,n_80=[160.]*2))
        for count,expected in [(5,.2),(10,(100*.2+10*.4)/110),(100,.4)]:
            current=prior.select('dataset','stat','level_id').with_columns(
                pl.lit(.4).alias('mu_current'),pl.lit(count).alias('n_players_current'))
            result=build_blended_rows(prior,current,2026,10,.8)
            self.assertEqual(result['K'].to_list(),[40.,40.])
            self.assertEqual(result['season'].to_list(),[2026.,2026.])
            for value in result['mu']:
                self.assertAlmostEqual(value,expected)


if __name__ == '__main__':
    unittest.main()

"""Counts, observed denominators, and role-specific rate/percentile presentation."""
import unittest
from unittest.mock import patch
import pandas as pd
import polars as pl

from app.swing_rates import add_swing_rates
from pipeline.swing_rates import add_swing_rates as add_polars_rates, swing_count_expressions
from app.table_layout import order_stat_columns, RAW_COUNT_COLUMNS
from app.utils import rank_for_display


class SwingRateLayoutTests(unittest.TestCase):
    def test_observed_counts_use_distinct_swings_and_pitch_denominators(self):
        data = pd.DataFrame({'rate_whiffs': [2, 0, 0], 'rate_swings': [5, 0, 2], 'rate_pitches': [10, 3, 5]})
        result = add_swing_rates(data)
        self.assertEqual(result.loc[0, ['whiff_pct', 'Swing_pct', 'SwStr']].tolist(), [40, 50, 20])
        self.assertTrue(pd.isna(result.loc[1, 'whiff_pct']))
        self.assertEqual(result.loc[1, 'SwStr'], 0)
        other = add_polars_rates(pl.from_pandas(data)).to_pandas()
        pd.testing.assert_frame_equal(result, other[result.columns], check_dtype=False)

    def test_compact_split_rates_use_equivalent_pooled_identities(self):
        result = add_swing_rates(pd.DataFrame({'Swing_pct': [50.], 'SwStr': [20.]}))
        self.assertEqual(result['whiff_pct'].iloc[0], 40)
        result = add_swing_rates(pd.DataFrame({'Swing_pct': [50.], 'whiff_pct': [40.]}))
        self.assertEqual(result['SwStr'].iloc[0], 20)

    def test_raw_pitch_flags_pool_before_division(self):
        pitches = pl.DataFrame({'team': ['A']*4, 'whiff': [1, 0, 0, 0], 'swing': [1, 1, 0, 0]})
        result = add_polars_rates(pitches.group_by('team').agg(swing_count_expressions(pitches)))
        self.assertEqual(result['whiff_pct'].item(), 50)
        self.assertEqual(result['SwStr'].item(), 25)

    def test_refresh_builders_retain_observed_denominators_for_every_scope(self):
        from pipeline import data_aggregate as aggregate
        from tests.test_combined_splits import CombinedSplitTests
        pitches = CombinedSplitTests().pitches().with_columns(pl.col('swing').alias('whiff'))
        for name in ['build_hitters', 'build_pitchers', 'build_pitch_types', 'build_team_hitting',
                     'build_team_pitching', 'build_league_hitting', 'build_league_pitching', 'build_league_pitch_types']:
            with self.subTest(builder=name):
                result = getattr(aggregate, name)(pitches)
                self.assertEqual(result['rate_pitches'].sum(), pitches.height)
                self.assertEqual(result['rate_swings'].sum(), pitches['swing'].sum())
                self.assertEqual(result['rate_whiffs'].sum(), pitches['whiff'].sum())
                measured = result.filter(pl.col('rate_swings') > 0)
                self.assertTrue((measured['whiff_pct'] == 100).all())

    def test_pitcher_order_keeps_counts_and_swing_rate_block_together(self):
        columns = ['Name', 'SwStr (%)', 'HR', 'Avg Miss Distance Pctile', 'BBE', 'Swing (%)', 'Pitches', 'Whiff (%)', 'pSwStr (%)', 'Other']
        result = order_stat_columns(columns, pitching=True)
        self.assertEqual(result[:4], ['Name', 'Pitches', 'BBE', 'HR'])
        i = result.index('SwStr (%)')
        self.assertEqual(result[i-2:i+1], ['Whiff (%)', 'Swing (%)', 'SwStr (%)'])
        self.assertEqual(result[result.index('pSwStr (%)')+1], 'Avg Miss Distance Pctile')
        self.assertEqual(set(result), set(columns))

    def test_hitter_order_places_tracking_and_rate_columns_at_their_anchors(self):
        columns = ['Name', 'Avg Swing Speed Pctile', 'Chase', 'Whiff vs Secondaries', 'Contact Over Expected', 'SwStr (%)', 'Avg Miss Distance Pctile', 'Whiff (%)', 'Swing (%)']
        result = order_stat_columns(columns)
        for before, after in [('Avg Miss Distance Pctile', 'Avg Swing Speed Pctile'), ('Whiff (%)', 'Whiff vs Secondaries'), ('Swing (%)', 'Chase'), ('Contact Over Expected', 'SwStr (%)')]:
            self.assertEqual(result.index(before)+1, result.index(after))

    def test_baserunning_counts_stay_with_takeoff_and_damage_follows_miss_distance(self):
        for pitching, takeoff in [(False, 'Takeoff%'), (True, 'Takeoff% Against')]:
            columns = ['Name', 'SB', 'BBE', 'Damage/BBE (%)', 'SBO', 'SwStr (%)', 'Avg Miss Distance (in.)', takeoff]
            result = order_stat_columns(columns, pitching=pitching)
            i = result.index(takeoff)
            self.assertEqual(result[i-2:i], ['SBO', 'SB'])
            if pitching:
                self.assertEqual(result.index('Avg Miss Distance (in.)')+1, result.index('Damage/BBE (%)'))

    def test_pitch_split_share_stays_between_pitch_count_and_batted_ball_counts(self):
        result = order_stat_columns(['Name', 'Pitch Type', 'HR', 'PA', 'BBE', '%', '#', 'Other'])
        self.assertEqual(result[:7], ['Name', 'Pitch Type', '#', '%', 'BBE', 'HR', 'PA'])

    def test_native_damage_rate_uses_batted_balls_and_keeps_missing_sample_blank(self):
        from pipeline.data_aggregate import _damage_rate_expressions
        pitch = pl.DataFrame({'is_in_play': [True, True, False], 'damage_pred': [95., 95., None],
            'exit_velo': [100., 80., None], 'launch_angle': [20., 20., None], 'spray_angle_adj': [0., 0., None]})
        result = pitch.select(_damage_rate_expressions()).row(0, named=True)
        self.assertEqual(result, {'Damage_pct': 50., 'Damage_pct_num': 1, 'Damage_pct_den': 2})
        self.assertIsNone(pitch.tail(1).select(_damage_rate_expressions())['Damage_pct'].item())

    def test_percentiles_invert_hitter_whiffs_only_once(self):
        frame = pd.DataFrame({'season': [2026]*3, 'whiff_pct': [10., 20., 30.], 'SwStr': [5., 10., 15.]})
        hitter = rank_for_display(frame, ['whiff_pct', 'SwStr'], ['season'], {'whiff_pct', 'SwStr'})
        pitcher = rank_for_display(frame, ['whiff_pct'], ['season'])
        self.assertGreater(hitter['whiff_pct_pctile'].iloc[0], hitter['whiff_pct_pctile'].iloc[2])
        self.assertGreater(hitter['SwStr_pctile'].iloc[0], hitter['SwStr_pctile'].iloc[2])
        self.assertLess(pitcher['whiff_pct_pctile'].iloc[0], pitcher['whiff_pct_pctile'].iloc[2])

    def test_counts_never_receive_conditional_backgrounds_even_with_page_overrides(self):
        from app.viz import render_table
        frame = pd.DataFrame({'Name': ['A','B','C'], 'BBE': [1,5,9], 'HR': [1,5,9], 'Pitches': [1,5,9], 'Whiff (%)': [10.,20.,30.]})
        for role in ['hitter', 'pitcher']:
            with patch('app.viz._is_user_subscribed', return_value=True), patch('app.viz._render_plot_controls') as plot, patch('app.viz.st.dataframe') as display:
                render_table(frame, show_controls=False, workload_kind=role, stats_df=frame, no_format_cols={'Unrelated'})
                styler = display.call_args.args[0]
                styles = styler._compute().ctx
                for col in ['BBE', 'HR', 'Pitches']:
                    index = styler.data.columns.get_loc(col)
                    self.assertTrue(all(not styles.get((row,index)) for row in range(3)))
                self.assertEqual('Whiff (%)' in plot.call_args.args[3], role=='hitter')

    def test_split_colors_follow_hitter_and_pitcher_meaning(self):
        from app.viz import render_table
        from app.hitter_pitch_splits import HITTING_REVERSE_COLUMNS
        frame = pd.DataFrame({'Name': ['A', 'B', 'C'], '__season': [2026]*3, '__level': [1]*3,
            'Whiff%': [10., 20., 30.], 'Avg Miss Distance (in.)': [1., 2., 3.],
            'Damage/BBE (%)': [10., 20., 30.], 'Chase (%)': [10., 20., 30.],
            '#': [100, 200, 300], 'BBE': [10, 20, 30], 'HR': [1, 2, 3]})
        for role in ['hitter', 'pitcher']:
            reverse = HITTING_REVERSE_COLUMNS if role == 'hitter' else {'Damage/BBE (%)'}
            with patch('app.viz._is_user_subscribed', return_value=True), patch('app.viz._render_plot_controls'), patch('app.viz.st.dataframe') as display:
                render_table(frame, workload_kind=role, reverse_cols=reverse, show_controls=False,
                             group_cols=['__season', '__level'], stats_df=frame)
                styler = display.call_args.args[0]
                styles = styler._compute().ctx
                for column in ['Whiff%', 'Avg Miss Distance (in.)', 'Damage/BBE (%)', 'Chase (%)']:
                    color = dict(styles[(0, styler.data.columns.get_loc(column))])['background-color']
                    red, green = [int(v) for v in color.split('(')[1].split(',')[:2]]
                    low_is_better = (column != 'Damage/BBE (%)') if role == 'hitter' else (column == 'Damage/BBE (%)')
                    self.assertEqual(green > red, low_is_better, (role, column))


if __name__ == '__main__':
    unittest.main()

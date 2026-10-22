"""Miss distance must use measured misses, never all-pitch or equal-player weights."""
import unittest
import pandas as pd
import polars as pl

from pipeline.tracking_metrics import miss_distance_expressions
from pipeline.backfill_miss_distance import merge_metric, aggregate_for
from pipeline.data_aggregate import _add_game_type_group, _tag_pitch, build_hitters, build_pitchers, build_pitch_types, build_team_hitting, build_team_pitching
from app.aggregates import aggregate_span, build_span_table, HITTER_SPEC, PITCHER_SPEC, PITCH_TYPE_SPEC, TEAM_HITTER_SPEC, TEAM_PITCHER_SPEC
import tests.test_combined_splits as split_fixtures
from pipeline.build_combined_splits import add_dimensions, build_intersection
from pipeline.build_gamelogs import build_hitter_gamelogs, build_pitcher_gamelogs, build_pitch_type_gamelogs


class MissDistanceTests(unittest.TestCase):
    def test_only_recorded_values_enter_the_mean(self):
        frame = pl.DataFrame({'team': ['A']*5, 'miss_distance': ['1', None, '9', 'NaN', 'bad']})
        row = frame.group_by('team').agg(miss_distance_expressions(frame)).row(0, named=True)
        self.assertEqual(row['miss_distance'], 5)
        self.assertEqual(row['miss_distance_n'], 2)

    def test_absent_source_column_stays_blank(self):
        frame = pl.DataFrame({'team': ['A', 'A']})
        row = frame.group_by('team').agg(miss_distance_expressions(frame)).row(0, named=True)
        self.assertIsNone(row['miss_distance'])
        self.assertEqual(row['miss_distance_n'], 0)

    def test_main_builders_retain_measured_counts(self):
        pitch = split_fixtures.CombinedSplitTests().pitches().with_columns(pl.Series('miss_distance', [1., 9., None, None, None, None]))
        pitch = _tag_pitch(_add_game_type_group(pitch))
        for builder in [build_hitters, build_pitchers, build_pitch_types, build_team_hitting, build_team_pitching]:
            with self.subTest(builder=builder.__name__):
                result = builder(pitch)
                self.assertEqual(result['miss_distance_n'].sum(), 2)
                measured = result.filter(pl.col('miss_distance_n') > 0)
                pooled = (measured['miss_distance'] * measured['miss_distance_n']).sum() / measured['miss_distance_n'].sum()
                self.assertEqual(pooled, 5)

    def test_date_ranges_weight_by_tracked_misses_in_every_role(self):
        for spec in [HITTER_SPEC, PITCHER_SPEC, PITCH_TYPE_SPEC, TEAM_HITTER_SPEC, TEAM_PITCHER_SPEC]:
            frame = pd.DataFrame({**{key: ['A', 'A'] for key in spec.group_keys},
                'game_pk': [1, 2], 'game_date': ['2026-05-01', '2026-05-02'],
                'PA': [100, 100], 'TBF': [100, 100], 'pitches': [100, 100],
                'miss_distance': [2., 10.], 'miss_distance_n': [1, 3]})
            for mode in ['Rates', 'Raw counts']:
                with self.subTest(role=spec.group_keys, mode=mode):
                    result = aggregate_span(frame, spec, mode)
                    self.assertEqual(result['Avg Miss Distance (in.)'].iloc[0], 8)
                    table = build_span_table(frame, spec, mode, 'Date', 'Range')
                    self.assertEqual(table['Avg Miss Distance (in.)'].tolist(), [8., 10., 2.])

    def test_join_keeps_order_and_existing_statistics(self):
        frame = pl.DataFrame({'team': ['B', 'A'], 'damage': [1/3, 2/3]})
        metrics = pl.DataFrame({'team': ['A'], 'miss_distance': [4.], 'miss_distance_n': [3]})
        result = merge_metric(frame, metrics, ['team'])
        self.assertTrue(result.select(frame.columns).equals(frame))
        self.assertEqual(result['miss_distance'].to_list(), [None, 4.])
        self.assertEqual(result['miss_distance_n'].to_list(), [0, 3])

    def test_combined_refresh_keeps_distance_and_its_measured_sample(self):
        pitch = split_fixtures.CombinedSplitTests().pitches().with_columns(
            pl.Series('miss_distance', [1., 9., None, None, None, None]))
        pitch = _tag_pitch(_add_game_type_group(pitch))
        for kind in ['hitter', 'pitcher', 'pitch', 'team_hitter', 'team_pitcher']:
            ready = add_dimensions(pitch, kind in {'hitter', 'team_hitter'})
            result = build_intersection(ready, ['split_hand', 'split_month'], kind, pl.DataFrame())
            self.assertIn('miss_distance_n', result.columns)
            self.assertEqual(result['miss_distance_n'].sum(), 2)

    def test_gamelog_refresh_retains_samples_for_later_team_and_date_pooling(self):
        pitch = split_fixtures.CombinedSplitTests().pitches().with_columns(
            pl.Series('miss_distance', [1., 9., None, None, None, None]),
            pl.lit(0).alias('bb'), pl.lit(0).alias('k'), pl.lit(False).alias('is_strike'))
        pitch = _tag_pitch(_add_game_type_group(pitch))
        for builder in [build_hitter_gamelogs, build_pitcher_gamelogs, build_pitch_type_gamelogs]:
            result = builder(pitch)
            self.assertEqual(result['miss_distance_n'].sum(), 2)


if __name__ == '__main__':
    unittest.main()

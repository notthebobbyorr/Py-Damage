"""The backfill repairs missing measurements without rewriting unrelated baseball data."""
import unittest
from unittest.mock import patch
from pathlib import Path
import tempfile
import polars as pl
from pipeline.backfill_bat_tracking import fill_measurements, replace_tracking, write_updated, ROOT
from pipeline.scrape_bat_tracking import FIELDS


class TrackingBackfillTests(unittest.TestCase):
    def sources(self):
        raw = pl.DataFrame({'game_pk': [10, 10], 'at_bat_index': [0, 1],
            'pitch_of_ab': [1, 1], 'batter_mlbid': [100, 101], 'pitcher_mlbid': [200, 200],
            'damage_pred': [80., 90.], **{c: [71., None] for c in FIELDS}})
        scrape = pl.DataFrame({'game_pk': [10, 10], 'at_bat_number': [1, 2],
            'pitch_number': [1, 1], 'batter': [100, 101], 'pitcher': [200, 200],
            **{c: [99., 75.] for c in FIELDS}})
        return raw, scrape

    def test_existing_values_survive_and_missing_values_are_repaired(self):
        raw, scrape = self.sources()
        result = fill_measurements(raw.lazy(), scrape).collect()
        self.assertEqual(result['bat_speed'].to_list(), [71., 75.])
        self.assertTrue(result.drop(FIELDS).equals(raw.drop(FIELDS)))

    def test_mismatched_batter_cannot_fill_another_hitters_pitch(self):
        raw, scrape = self.sources()
        scrape = scrape.with_columns(pl.lit(999).alias('batter'))
        self.assertIsNone(fill_measurements(raw.lazy(), scrape).collect()['bat_speed'][1])

    def test_mismatched_pitcher_cannot_fill_a_substitution(self):
        raw, scrape = self.sources()
        scrape = scrape.with_columns(pl.lit(999).alias('pitcher'))
        self.assertIsNone(fill_measurements(raw.lazy(), scrape).collect()['bat_speed'][1])

    def test_duplicate_scraper_pitches_fail_before_joining(self):
        raw, scrape = self.sources()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            fill_measurements(raw.lazy(), pl.concat([scrape, scrape]))

    def test_only_tracking_statistics_change_in_aggregate_tables(self):
        frame = pl.DataFrame({'team': ['A', 'B'], 'bat_speed': [None, 70.], 'PA': [100, 200]})
        metrics = pl.DataFrame({'team': ['B', 'A'], 'bat_speed': [72., 74.]})
        result = replace_tracking(frame, metrics, ['team'])
        self.assertEqual(result['bat_speed'].to_list(), [74., 72.])
        self.assertEqual(result['PA'].to_list(), [100, 200])

    def test_ambiguous_aggregate_join_fails(self):
        frame = pl.DataFrame({'team': ['A'], 'bat_speed': [None]}, schema_overrides={'bat_speed': pl.Float64})
        metrics = pl.DataFrame({'team': ['A', 'A'], 'bat_speed': [71., 72.]})
        with self.assertRaises(pl.exceptions.ComputeError):
            replace_tracking(frame, metrics, ['team'])

    def test_split_writer_preserves_precision_and_nan_values(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            output = Path(folder) / 'output'
            output.mkdir()
            path = output / 'hitter_splits.parquet'
            frame = pl.DataFrame({'split': ['May', 'June'], 'chase': [100/3, float('nan')],
                                  'bat_speed': [72.1, 73.2]})
            with patch('pipeline.backfill_bat_tracking.OUT', output):
                write_updated(frame, path, Path(folder) / 'backup')
            self.assertTrue(pl.read_parquet(path).equals(frame))


if __name__ == '__main__':
    unittest.main()

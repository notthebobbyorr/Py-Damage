import unittest
import polars as pl

from pipeline.data_aggregate import bat_tracking_expressions
from pipeline.backfill_hitting_bat_tracking import merge_tracking


class HittingBatTrackingTests(unittest.TestCase):
    def test_each_metric_uses_its_own_recorded_sample_and_fast_swing_includes_75(self):
        pitch = pl.DataFrame({"season": [2026]*4, "bat_speed": ["70", "75", "80", None],
                              "swing_length": [None, "6", None, "8"]})
        row = pitch.group_by("season").agg(bat_tracking_expressions(pitch)).row(0, named=True)
        self.assertEqual(row["bat_speed"], 75)
        self.assertEqual(row["bat_speed_n"], 3)
        self.assertAlmostEqual(row["fast_swing_pct"], 200/3)
        self.assertEqual(row["swing_length"], 7)
        self.assertEqual(row["swing_length_n"], 2)
        self.assertIsNone(row["attack_angle"])
        self.assertEqual(row["attack_angle_n"], 0)

    def test_backfill_preserves_existing_rows_values_and_missing_tracking(self):
        existing = pl.DataFrame({"season": [2025, 2026], "PA": [100, 200]})
        tracking = pl.DataFrame({"season": [2026], "bat_speed": [75.]})
        result = merge_tracking(existing, tracking, ["season"])
        self.assertEqual(result.select(existing.columns).to_dicts(), existing.to_dicts())
        self.assertEqual(result["bat_speed"].to_list(), [None, 75.])
        with self.assertRaises(ValueError):
            merge_tracking(existing, pl.concat([tracking, tracking]), ["season"])


if __name__ == "__main__":
    unittest.main()

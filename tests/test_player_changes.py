import unittest
import pandas as pd
from app.changes import compare_periods, player_profile
from app.aggregates import aggregate_span, build_span_table, PITCH_TYPE_SPEC


class ChangesTests(unittest.TestCase):
    def test_comparison_requires_both_samples_and_same_level(self):
        before = pd.DataFrame({"id": [1, 2, 3], "level_id": [1, 1, 1], "name": ["A", "B", "C"], "metric": [10, 20, 30], "PA": [100, 99, 100]})
        after = before.copy()
        after["metric"] = [15, 25, 35]
        after.loc[2, "level_id"] = 11
        before["id"] = before["id"].astype("Int32")
        after["id"] = after["id"].astype("int64")
        result = compare_periods(before, after, "id", "name", "metric", "PA", 100)
        self.assertEqual(result["Player ID"].tolist(), [1])
        self.assertEqual(result["Change"].tolist(), [5])

    def test_player_profile_selects_player_and_preserves_missing_values(self):
        before = pd.DataFrame({"id": [1, 2], "level_id": [1, 1], "name": ["A", "B"], "PA": [100, 200], "speed": [90., 110.], "contact": [80., 99.]})
        after = before.copy()
        before["HR"] = pd.Series([10, 20], dtype="UInt64")
        after["HR"] = pd.Series([5, 25], dtype="UInt64")
        after["speed"] = [92., 120.]
        after.loc[0, "contact"] = float("nan")
        result = player_profile(before, after, 1, "id", "name", {"speed": "Speed", "contact": "Contact (%)"}, "PA", 100).set_index("Metric")
        self.assertEqual(result.loc["Speed", "Change"], 2)
        self.assertEqual(result.loc["PA", "Before"], 100)
        self.assertEqual(result.loc["HR", "Change"], -5)
        import pyarrow as pa
        pa.Table.from_pandas(result)
        self.assertEqual(result.loc["Contact (%)", "Before"], 80)
        self.assertTrue(pd.isna(result.loc["Contact (%)", "After"]))
        self.assertTrue(pd.isna(result.loc["Contact (%)", "Change"]))
        self.assertTrue(player_profile(before, after, 1, "id", "name", {"speed": "Speed"}, "PA", 101).empty)
        self.assertTrue(player_profile(before, after[after.id == 2], 1, "id", "name", {"speed": "Speed"}, "PA", 0).empty)

    def test_team_codes_match_without_numeric_player_ids(self):
        before = pd.DataFrame({"__team_code": ["BOS", "NYY"], "__team_name": ["BOS", "NYY"], "level_id": [1, 1], "PA": [200, 200], "metric": [10., 20.]})
        after = before.copy()
        after["metric"] = [15., 18.]
        result = compare_periods(before, after, "__team_code", "__team_name", "metric", "PA", 100)
        self.assertEqual(result["Change"].tolist(), [5., -2.])
        profile = player_profile(before, after, "BOS", "__team_code", "__team_name", {"metric": "Metric"}, "PA", 100)
        self.assertEqual(profile.iloc[-1]["Change"], 5.)

    def test_half_season_cutoff_and_unknown_year(self):
        import polars as pl
        from pipeline.data_aggregate import _split_half
        df = pl.DataFrame({"season": [2026]*3, "game_date": ["2026-07-13", "2026-07-14", "2026-07-15"]})
        self.assertEqual(_split_half(df)["__split"].to_list(), ["1st Half", None, "2nd Half"])
        with self.assertRaisesRegex(ValueError, "Missing All-Star cutoff"):
            _split_half(df.with_columns(pl.lit(2099).alias("season")))

    def test_traits_weight_only_measured_pitches(self):
        df = pd.DataFrame({"pitcher_mlbid": [1, 1], "pitch_tag": ["CH", "CH"], "pitches": [100, 100], "rpm": [2000, 1000], "rpm_n": [10, 30]})
        result = aggregate_span(df, PITCH_TYPE_SPEC, "Rates")
        self.assertEqual(result["RPM"].iloc[0], 1250)

    def test_date_range_traits_in_both_modes_round_after_weighting(self):
        df = pd.DataFrame({
            "pitcher_mlbid": [1, 1], "pitch_tag": ["CH", "CH"],
            "pitches": [10, 30], "velo": [80.0, 84.0],
            "vbreak": [10.0, 14.0], "vbreak_n": [10, 30],
            "hbreak": [-5.0, -9.0], "hbreak_n": [10, 30],
            "rpm": [2000.4, 1000.4], "rpm_n": [10, 30],
        })
        for mode in ["Raw counts", "Rates"]:
            table = build_span_table(df, PITCH_TYPE_SPEC, mode, "Date", "TOTAL")
            self.assertEqual(table.loc[0, "Avg mph"], 83.0)
            self.assertEqual(table.loc[0, "IVB (in.)"], 13.0)
            self.assertEqual(table.loc[0, "HB (in.)"], -8.0)
            self.assertEqual(table["RPM"].tolist(), [1250, 2000, 1000])
            self.assertEqual(str(table["RPM"].dtype), "Int64")

    def test_ambiguous_duplicate_metrics_are_excluded(self):
        df = pd.DataFrame({"id": [1, 1], "level_id": [1, 1], "name": ["A", "A"], "metric": [10, 20], "PA": [100, 100]})
        self.assertTrue(compare_periods(df, df, "id", "name", "metric", "PA", 0).empty)

    def test_split_rows_recombine_using_rate_denominators(self):
        df = pd.DataFrame({"id": [1, 1], "level_id": [1, 1], "name": ["A", "A"], "metric": [10, 50], "metric_num": [1, 15], "metric_den": [10, 30], "PA": [40, 60]})
        result = compare_periods(df, df, "id", "name", "metric", "PA", 100)
        self.assertEqual(result["Before"].tolist(), [40])
        self.assertEqual(result["Before PA"].tolist(), [100])

if __name__ == "__main__":
    unittest.main()

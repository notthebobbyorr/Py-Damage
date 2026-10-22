import unittest
import pandas as pd

from app.hitting_pitch_split_tables import prepare_hitter_pitch_splits


class HitterPitchLeaderboardTests(unittest.TestCase):
    def test_metadata_matches_level_and_season_without_multiplying_rows(self):
        source = pd.DataFrame({"batter_mlbid": [1, 1, 2], "season": [2026]*3,
                               "level_id": [1, 11, 1], "pitch_group": ["FA"]*3,
                               "pitches": [100, 50, 10], "pitch_pct": [40., 30., 20.]})
        metadata = pd.DataFrame({"batter_mlbid": [1]*5, "season": [2026]*4 + [2025],
                                 "level_id": [1, 1, 11, 1, 1],
                                 "hitter_name": ["Test", "Test Alias", "Test AAA", "Postseason", "Last Year"],
                                 "hitting_code": ["AAA", "BBB", "CCC", "DDD", "EEE"],
                                 "game_type_group": ["Regular Season"]*3 + ["Postseason", "Regular Season"]})
        result = prepare_hitter_pitch_splits(source, metadata)
        self.assertEqual(len(result), len(source))
        self.assertEqual(result.hitting_code.tolist(), ["AAA | BBB", "CCC", ""])
        self.assertEqual(result.hitter_name.tolist(), ["Test", "Test AAA", "2"])
        pd.testing.assert_frame_equal(result[source.columns], source)


if __name__ == "__main__":
    unittest.main()

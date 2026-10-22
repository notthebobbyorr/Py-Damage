import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
import polars as pl

from pipeline.pitch_split_games import apply_game_classifications, verified_games
from pipeline.build_hitter_pitch_splits import build_splits


class PitchSplitGameTests(unittest.TestCase):
    def test_resumed_game_duplicates_are_cached_once_without_repeated_requests(self):
        game = {"gamePk": 1, "gameType": "R", "teams": {
            side: {"team": {"sport": {"id": 11}}} for side in ["home", "away"]}}
        session = MagicMock()
        session.__enter__.return_value = session
        session.get.return_value.json.return_value = {"dates": [{"games": [game]}, {"games": [game]}]}
        scan = pl.DataFrame({"game_pk": [1], "level_id": [1], "hitting_code": ["234"]}).lazy()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder, patch("pipeline.pitch_split_games.requests.Session", return_value=session):
            cache = Path(folder) / "games.parquet"
            self.assertEqual(len(verified_games([scan], cache)), 1)
            self.assertEqual(len(verified_games([scan], cache)), 1)
            self.assertEqual(session.get.call_count, 1)

    def test_game_metadata_corrects_entire_game_without_multiplying_pitches(self):
        pitch = pl.DataFrame({"game_pk": [1, 1, 2, 3], "level_id": [1]*4,
                              "game_type": ["R"]*4})
        verified = pl.DataFrame({"game_pk": [1, 2], "verified_level": [11, 1],
                                 "verified_type": ["R", "S"]})
        result = apply_game_classifications(pitch.lazy(), verified).collect()
        self.assertEqual(result["level_id"].to_list(), [11, 11, 1, 1])
        self.assertEqual(result["game_type"].to_list(), ["R", "R", "S", "R"])
        self.assertEqual(len(result), len(pitch))

    def test_unknown_and_missing_types_cannot_be_assumed_regular_season(self):
        for scope in ["hitter", "team", "league"]:
            with self.subTest(scope=scope):
                with self.assertRaises(ValueError):
                    build_splits(pl.DataFrame({"game_pk": [1]}), scope)
                for game_type in ["S", "F", "D", "L", "W", "X", None]:
                    self.assertTrue(build_splits(pl.DataFrame({"game_type": [game_type]}), scope).is_empty())


if __name__ == "__main__":
    unittest.main()

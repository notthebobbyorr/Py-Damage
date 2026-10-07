import unittest
import polars as pl
from pipeline.backfill_position_games import count_position_games

class PositionGamesTests(unittest.TestCase):
    def test_repeated_pitches_and_game_types(self):
        raw=pl.DataFrame(dict(batter_mlbid=[1]*6,level_id=[1]*6,season=[2026]*6,
            game_type=['R','R','R','R','D','D'],game_pk=[10,10,11,11,12,12],
            batter_position=[6,6,6,4,6,6]))
        result=count_position_games(raw).set_index('game_type_group')
        self.assertEqual(result.loc['Regular Season','SS'],2)
        self.assertEqual(result.loc['Regular Season','X2B'],1)
        self.assertEqual(result.loc['Postseason','SS'],1)
        self.assertEqual(result.loc['Postseason','X2B'],0)

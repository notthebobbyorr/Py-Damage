import unittest
import pandas as pd
from pipeline.backfill_pitcher_arsenal import build_arsenal

class ArsenalTests(unittest.TestCase):
    def test_primary_type_usage_and_missing_groups(self):
        rows=[]
        for hand,tag,count,velo in [('R','FA',100,95),('R','SL',60,85),('R','CU',40,78),('R','CH',20,88),('L','FA',10,92)]:
            rows.append(dict(pitcher_mlbid=1,season=2026,level_id=11,game_type_group='Regular Season',pitcher_hand=hand,name='Pitcher',pitch_tag=tag,pitches=count,velo=velo))
        result=build_arsenal(pd.DataFrame(rows)).set_index('pitcher_hand')
        self.assertAlmostEqual(result.loc['R','BB_pct'],100*100/220)
        self.assertAlmostEqual(result.loc['R','OFF_pct'],100*20/220)
        self.assertEqual(result.loc['R','BB_velo'],85)
        self.assertEqual(result.loc['R','OFF_velo'],88)
        self.assertEqual(result.loc['L','BB_pct'],0)
        self.assertTrue(pd.isna(result.loc['L','BB_velo']))
    def test_tie_is_deterministic_and_missing_velocity_is_not_replaced(self):
        rows=[dict(pitcher_mlbid=1,season=2026,level_id=1,game_type_group='Regular Season',pitcher_hand='R',name='P',pitch_tag=tag,pitches=50,velo=velo) for tag,velo in [('SL',85),('CU',float('nan'))]]
        self.assertTrue(pd.isna(build_arsenal(pd.DataFrame(rows)).iloc[0].BB_velo))

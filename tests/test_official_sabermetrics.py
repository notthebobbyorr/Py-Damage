import unittest
from unittest.mock import Mock
import pandas as pd
from pipeline.official_sabermetrics import fetch_sabermetric
from app.official_teams import detail_columns
from app.official_hitting import DETAIL_COLUMNS as HITTER_COLUMNS
from app.official_pitching import DETAIL_COLUMNS as PITCHER_COLUMNS


class SabermetricTests(unittest.TestCase):
    def test_combined_traded_total_is_selected(self):
        client = Mock()
        rows = [dict(season='2025', player={'id': 1}, numTeams=n, stat={'wRcPlus': v})
                for n, v in [(1, 90), (2, 110)]]
        client._get.return_value = ({'stats': [{'group': {'displayName': 'hitting'},
            'totalSplits': 2, 'splits': rows}]}, None)
        frame = fetch_sabermetric(client, 2025, 1, 'hitting')
        self.assertEqual(frame['wRC+'].tolist(), [110])

    def test_missing_minor_league_values_and_team_columns(self):
        client = Mock()
        client._get.return_value = ({'stats': []}, None)
        self.assertTrue(fetch_sabermetric(client, 2025, 11, 'pitching').empty)
        self.assertNotIn('wRC+', detail_columns())
        self.assertNotIn('xFIP', detail_columns(True))
        self.assertNotIn('RA9', PITCHER_COLUMNS)
        self.assertEqual(HITTER_COLUMNS[HITTER_COLUMNS.index('OPS') + 1], 'wRC+')
        self.assertEqual(PITCHER_COLUMNS[PITCHER_COLUMNS.index('ERA') + 1], 'xFIP')

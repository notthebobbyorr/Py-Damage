import unittest
from unittest.mock import patch, Mock

import pandas as pd

from pipeline.build_official_pitching import FIELDS, normalize_totals, fetch_advanced
from app.official_pitching import DETAIL_COLUMNS, detail_values
from app.official_hitting import render_official_table


class OfficialPitchingTests(unittest.TestCase):
    def source(self):
        row = dict.fromkeys(FIELDS, 0)
        row.update(player_id=1, season=2026, player_name="Test", innings_pitched="6.2",
                   wins=2, losses=1, era="2.70", whip="1.05", avg=".210",
                   batters_faced=100, strike_outs=30, base_on_balls=8)
        return pd.DataFrame([row])

    def test_innings_record_and_rates(self):
        values = dict(zip(DETAIL_COLUMNS, detail_values(normalize_totals(self.source()).iloc[0])))
        self.assertEqual(values["IP"], "6.2")
        self.assertEqual(values["W-L"], "2-1")
        self.assertEqual([values[c] for c in ["K%", "BB%", "K-BB%"]], ["30.0%", "8.0%", "22.0%"])
        self.assertEqual(values["ERA"], "2.70")
        self.assertEqual(values["AVG"], ".210")

    def test_relief_percentage_and_extra_rates(self):
        source = self.source().assign(inherited_runners=10, inherited_runners_scored=3,
                                     quality_starts=2, babip=".289", ops=".650", holds=4)
        values = dict(zip(DETAIL_COLUMNS, detail_values(normalize_totals(source).iloc[0])))
        self.assertEqual(values["Inh. Runners %"], "30.0%")
        self.assertEqual(values["QS"], "2")
        self.assertEqual(values["BABIP"], ".289")
        self.assertEqual(values["OPS"], ".650")
        self.assertEqual(values["Hld"], "4")
        zero = normalize_totals(self.source()).iloc[0]
        self.assertTrue(pd.isna(zero["Inh. Runners %"]))

    def test_advanced_pagination_and_missing_values(self):
        def page(player_id, stat):
            return {"stats": [{"group": {"displayName": "pitching"}, "totalSplits": 2,
                               "splits": [{"season": "2026", "sport": {"id": 11},
                                           "player": {"id": player_id}, "stat": stat}]}]}, None
        client = Mock()
        client._get.side_effect = [page(1, {"qualityStarts": 3, "babip": ".300"}), page(2, {})]
        result = fetch_advanced(client, 2026, 11).set_index("player_id")
        self.assertEqual(result.loc[1, "quality_starts"], 3)
        self.assertTrue(pd.isna(result.loc[2, "quality_starts"]))
        self.assertIn("offset=1", client._get.call_args.args[0])

    def test_advanced_repeated_page_is_rejected(self):
        client = Mock()
        client._get.return_value = ({"stats": [{"group": {"displayName": "pitching"}, "totalSplits": 2,
                                               "splits": [{"season": "2026", "player": {"id": 1},
                                                           "stat": {"qualityStarts": 3}}]}]}, None)
        with self.assertRaises(ValueError):
            fetch_advanced(client, 2026, 1)

    def test_missing_source_rate(self):
        result = normalize_totals(self.source().assign(era="-"))
        self.assertTrue(pd.isna(result.iloc[0].ERA))

    def test_zero_batters_and_invalid_innings(self):
        result = normalize_totals(self.source().assign(batters_faced=0))
        self.assertTrue(result[["K%", "BB%", "K-BB%"]].isna().all().all())
        with self.assertRaises(ValueError):
            normalize_totals(self.source().assign(innings_pitched="6.3"))

    @patch("app.official_hitting._component")
    @patch("app.official_pitching.load_official_pitching")
    def test_original_level_and_postseason_matching(self, load, component):
        load.return_value = normalize_totals(self.source(), 11).set_index(["player_id", "season", "level_id"])
        full = pd.DataFrame({"Player ID": [1, 1], "Season": [2026, 2026], "__level": [1, 1],
                             "__official_level": [11, 11], "__official_game_type": ["Regular Season", "Postseason"]})
        render_official_table(pd.DataFrame({"Name": ["Test", "Test"]}), full, "pitchers", pitching=True)
        data = component.call_args.kwargs["data"]
        self.assertEqual(data["columns"], DETAIL_COLUMNS)
        self.assertEqual(data["details"][0][2], "6.2")
        self.assertIsNone(data["details"][1])

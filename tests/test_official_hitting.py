import unittest
from unittest.mock import patch, Mock
from types import SimpleNamespace

import pandas as pd

from pipeline.build_official_hitting import FIELDS, normalize_totals, fetch_season
from app.official_hitting import detail_values, render_official_table


class OfficialHittingTests(unittest.TestCase):
    def source(self):
        row = {key: 0 for key in FIELDS}
        row.update(player_id=592450, player_name="Test", season=2026,
                   plate_appearances=285, strike_outs=85, base_on_balls=43,
                   avg=".241", obp=".360", slg=".511", ops=".871", babip=".289")
        return pd.DataFrame([row])

    def test_rates_use_pa_and_slash_keeps_three_places(self):
        row = normalize_totals(self.source()).iloc[0]
        self.assertEqual(detail_values(row)[5:11], [".241/.360/.511", ".871", "—", ".289", "29.8%", "15.1%"])

    def test_zero_pa_is_missing_rate_not_zero_percent(self):
        source = self.source()
        source["plate_appearances"] = 0
        self.assertEqual(detail_values(normalize_totals(source).iloc[0])[9:11], ["—", "—"])

    def test_undefined_babip_is_not_zero(self):
        source = self.source().assign(babip=".---")
        values = detail_values(normalize_totals(source).iloc[0])
        self.assertEqual(values[8], "—")

    @patch("app.official_hitting._component")
    @patch("app.official_hitting.load_official_hitting")
    def test_promoted_player_gets_separate_level_totals(self, load, component):
        mlb = normalize_totals(self.source(), 1)
        aaa = normalize_totals(self.source().assign(plate_appearances=100), 11)
        load.return_value = pd.concat([mlb, aaa]).set_index(["player_id", "season", "level_id"])
        full = pd.DataFrame({"Player ID": [592450]*2, "Season": [2026]*2, "__level": [11, 1]})
        render_official_table(pd.DataFrame({"Name": ["Test"]*2}), full, "levels")
        details = component.call_args.kwargs["data"]["details"]
        self.assertEqual([row[1] for row in details], ["100", "285"])

    @patch("app.official_hitting._component")
    @patch("app.official_hitting.load_official_hitting")
    def test_comps_use_original_level_and_never_attach_regular_totals_to_postseason(self, load, component):
        mlb = normalize_totals(self.source(), 1)
        aaa = normalize_totals(self.source().assign(plate_appearances=100), 11)
        load.return_value = pd.concat([mlb, aaa]).set_index(["player_id", "season", "level_id"])
        full = pd.DataFrame({
            "Player ID": [592450]*2, "Season": [2026]*2, "__level": [1, 1],
            "__official_level": [11, 1],
            "__official_game_type": ["Regular Season", "Postseason"],
        })
        render_official_table(pd.DataFrame({"Name": ["Test"]*2}), full, "comps")
        details = component.call_args.kwargs["data"]["details"]
        self.assertEqual(details[0][1], "100")
        self.assertIsNone(details[1])

    def test_duplicate_player_seasons_fail_instead_of_double_counting(self):
        with self.assertRaises(ValueError):
            normalize_totals(pd.concat([self.source(), self.source()]))

    def test_traded_player_uses_combined_total_without_summing_splits(self):
        source = self.source()
        total = source.assign(num_teams=2)
        client = Mock()
        client._get.return_value = ({"stats": []}, None)
        client.season_stats.side_effect = [
            SimpleNamespace(tables={"season_stats": pd.concat([source, source])}),
            SimpleNamespace(tables={"season_stats": pd.concat([source.assign(num_teams=1), total])}),
        ]
        result = fetch_season(client, 2026)
        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0].PA, 285)

    @patch("app.official_hitting._component")
    @patch("app.official_hitting.load_official_hitting")
    def test_row_identity_survives_order_and_does_not_match_minor_leagues(self, load, component):
        load.return_value = normalize_totals(self.source()).set_index(["player_id", "season", "level_id"])
        full = pd.DataFrame({"Player ID": [592450, 592450, 999], "Season": [2026]*3, "__level": [11, 1, 1]})
        display = pd.DataFrame({"Name": ["Minor", "<script>bad</script>", "Missing"], "Value": [1.2]*3})
        render_official_table(display.style.format({"Value": "{:.1f}"}), full, "test")
        data = component.call_args.kwargs["data"]
        self.assertIsNone(data["details"][0])
        self.assertEqual(data["details"][1][5], ".241/.360/.511")
        self.assertIsNone(data["details"][2])
        self.assertNotIn("<script>", data["html"])
        self.assertIn(">1.2<", data["html"])

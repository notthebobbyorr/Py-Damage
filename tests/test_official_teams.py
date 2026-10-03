import unittest
from unittest.mock import Mock, patch

import pandas as pd

from pipeline.build_official_teams import normalize_team, fetch_stats, HITTING_FIELDS, PITCHING_FIELDS
from app.official_teams import detail_columns, detail_values, load_official_teams
from app.official_hitting import render_official_table


class OfficialTeamTests(unittest.TestCase):
    def source(self, pitching=False):
        row = dict.fromkeys(PITCHING_FIELDS if pitching else HITTING_FIELDS, 0)
        row.update(team_id=147, team_name="Yankees", season=2026, runs=5)
        if pitching:
            row.update(innings_pitched="6.2", inherited_runners=None, inherited_runners_scored=None)
        else:
            row.update(games_played=2)
        return pd.DataFrame([row])

    def test_rates_use_team_games_and_baseball_outs(self):
        hitting = normalize_team(self.source(), 1, "hitting").iloc[0]
        pitching = normalize_team(self.source(True), 1, "pitching").iloc[0]
        self.assertEqual(hitting["R/G"], 2.5)
        self.assertEqual(pitching["RA9"], 6.75)
        values = dict(zip(detail_columns(True), detail_values(pitching, True)))
        self.assertEqual(values["RA9"], "6.75")
        self.assertNotIn("Inh. Runners %", values)
        hit_values = dict(zip(detail_columns(), detail_values(hitting)))
        self.assertEqual(list(hit_values)[4:8], ["HR", "R", "R/G", "AVG/OBP/SLG"])
        self.assertEqual([hit_values["R"], hit_values["R/G"]], ["5", "2.50"])
        self.assertIn("Hld", values)

    def test_no_sample_is_missing(self):
        hitting = normalize_team(self.source().assign(games_played=0), 1, "hitting").iloc[0]
        pitching = normalize_team(self.source(True).assign(innings_pitched="0.0"), 1, "pitching").iloc[0]
        self.assertTrue(pd.isna(hitting["R/G"]))
        self.assertTrue(pd.isna(pitching["RA9"]))

    @patch("app.official_hitting._component")
    @patch("app.official_teams.load_official_teams")
    def test_team_season_level_and_postseason_matching(self, load, component):
        frame = normalize_team(self.source(), 1, "hitting").assign(team_code="ATH")
        load.return_value = frame.set_index(["team_code", "season", "level_id"])
        full = pd.DataFrame({"Team": ["OAK"] * 4, "Season": [2026, 2025, 2026, 2026],
                             "__level": [1, 1, 11, 1],
                             "__official_game_type": ["Regular Season"] * 3 + ["Postseason"]})
        render_official_table(full[["Team", "Season"]], full, "teams", team=True)
        details = component.call_args.kwargs["data"]["details"]
        self.assertIsNotNone(details[0])
        self.assertEqual(details[1:], [None, None, None])

    def test_repeated_api_page_fails(self):
        client = Mock()
        client._get.return_value = ({"stats": [{"group": {"displayName": "hitting"},
            "totalSplits": 2, "splits": [{"season": "2026", "team": {"id": 1}}]}]}, None)
        with self.assertRaises(ValueError):
            fetch_stats(client, 2026, 1, "hitting", "season")

    @patch("app.official_teams.pd.read_parquet")
    @patch("app.official_teams.Path.exists", return_value=True)
    def test_ambiguous_team_codes_do_not_attach_wrong_totals(self, exists, read):
        read.return_value = pd.DataFrame({"group": ["hitting"] * 3,
            "team_code": ["COL", "COL", "NYY"], "season": [2015] * 3,
            "level_id": [11, 11, 1], "team_id": [1, 2, 147]})
        load_official_teams.clear()
        frame = load_official_teams()
        self.assertEqual(list(frame.index), [("NYY", 2015, 1)])
        load_official_teams.clear()

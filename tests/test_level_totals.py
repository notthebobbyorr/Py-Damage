import unittest

import pandas as pd

from app.aggregates import TEAM_HITTER_SPEC, build_span_table


class LevelTotalsTests(unittest.TestCase):
    def test_all_levels_keep_team_totals_separate(self):
        rows = pd.DataFrame({
            "hitting_code": ["TEAM", "TEAM"],
            "level_id": [1, 11],
            "game_pk": [1, 2],
            "game_date": ["2026-09-01", "2026-09-02"],
            "PA": [40, 50],
        })
        table = build_span_table(rows, TEAM_HITTER_SPEC, "Raw counts", "Date", "TOTAL")
        totals = table[table["Date"] == "TOTAL"].set_index("level_id")
        self.assertEqual(totals["PA"].to_dict(), {1: 40, 11: 50})
        self.assertEqual(totals["G"].to_dict(), {1: 1, 11: 1})

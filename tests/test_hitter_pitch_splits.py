import unittest
from unittest.mock import patch

import pandas as pd
import polars as pl

from app.hitter_pitch_splits import COLUMNS, split_values
from app.official_hitting import render_official_table
from pipeline.build_hitter_pitch_splits import SOURCE_COLUMNS, build_splits


class HitterPitchSplitTests(unittest.TestCase):
    def test_unassigned_team_pitches_remain_in_league_totals(self):
        pitch = self.pitches().with_columns(pl.lit(None, dtype=pl.String).alias("hitting_code"))
        self.assertTrue(build_splits(pitch, "team").is_empty())
        league = build_splits(pitch, "league").filter(pl.col("level_id") == 1)
        self.assertEqual(league.filter(pl.col("pitch_group") == "FA")["pitches"].item(), 3)

    def test_team_and_league_pool_pitches_and_keep_traded_player_teams_separate(self):
        records = self.pitches().to_dicts()
        records[1]["hitting_code"] = "BBB"
        for index, ev in [(0, 80), (1, 100), (9, 120)]:
            records[index].update(exit_velo=ev, launch_angle=10, bbe=1, is_in_play=True)
        pitch = pl.DataFrame(records)
        teams = build_splits(pitch, "team").filter(pl.col("level_id") == 1)
        league = build_splits(pitch, "league").filter(pl.col("level_id") == 1)
        fa = league.filter(pl.col("pitch_group") == "FA").row(0, named=True)
        team_fa = teams.filter(pl.col("pitch_group") == "FA")
        self.assertEqual(team_fa["pitches"].sum(), fa["pitches"])
        self.assertEqual(fa["pitches"], 3)
        self.assertAlmostEqual(fa["whiff_pct"], 200 / 3)
        self.assertNotAlmostEqual(fa["whiff_pct"], team_fa["whiff_pct"].mean())
        self.assertEqual(fa["EV90th"], 120)
        self.assertNotEqual(fa["EV90th"], team_fa["EV90th"].mean())
        self.assertAlmostEqual(fa["damage_rate"], 200 / 3)
        self.assertAlmostEqual(fa["pitch_pct"], 100 / 3)
        self.assertEqual(team_fa.filter(pl.col("hitting_code") == "BBB")["pitches"].item(), 1)

    def test_team_aliases_are_combined_before_pitch_shares_are_calculated(self):
        pitch = self.pitches().with_columns(
            pl.when(pl.col("batter_mlbid") == 1).then(pl.lit("OAK"))
            .otherwise(pl.lit("ATH")).alias("hitting_code")
        )
        teams = build_splits(pitch, "team").filter(pl.col("level_id") == 1)
        self.assertEqual(teams["hitting_code"].unique().to_list(), ["ATH"])
        fa = teams.filter(pl.col("pitch_group") == "FA").row(0, named=True)
        self.assertEqual(fa["pitches"], 3)
        self.assertAlmostEqual(fa["pitch_pct"], 100 / 3)

    def pitches(self):
        base = dict.fromkeys(SOURCE_COLUMNS, 0)
        base.update(
            hitter_name="Jos\u00e9 Test", batter_mlbid=1, season=2026, level_id=1,
            game_type="R", batter_hand="R", hitting_code="AAA", pitch_outcome="S",
            batter_position=3, is_in_play=False, is_inzone_pi=True, decision_value=1,
            pred_whiff_loc=0.4, damage_pred=90, exit_velo=None, launch_angle=None,
        )
        rows = []
        for i, (group, pitch_type, swing, whiff) in enumerate([
            ("FA", "FA", 1, 1), ("FA", "FC", 1, 0), ("SL", "SL", 0, 0),
            ("CH", "CH", 1, 1), ("XX", "XX", 0, 0), ("FA", "SI", 1, 0),
            ("FA", "FT", 1, 1), ("SL", "FC", 0, 0),
        ]):
            rows.append(dict(base, pi_pitch_group=group, pi_pitch_type=pitch_type,
                             swing=swing, whiff=whiff, pa_id=i, game_pk=1))
        # These pitches must not affect the first player's regular-season MLB shares.
        rows += [dict(rows[0], game_type="F"), dict(rows[0], batter_mlbid=2),
                 dict(rows[0], level_id=11)]
        rows[1]["hitter_name"] = "Joe Test"
        return pl.DataFrame(rows).with_columns(
            pl.col("exit_velo", "launch_angle").cast(pl.Float64)
        )

    def test_counts_include_unknown_in_denominator_and_whiffs_use_swings(self):
        result = build_splits(self.pitches())
        player = result.filter((pl.col("batter_mlbid") == 1) & (pl.col("level_id") == 1))
        rows = {r["pitch_group"]: r for r in player.to_dicts()}
        self.assertEqual(rows["FA"]["pitches"], 2)
        self.assertEqual(rows["FA"]["pitch_pct"], 25)
        self.assertEqual(rows["FA"]["Swing_pct"], 100)
        self.assertEqual(rows["FA"]["whiff_pct"], 50)
        self.assertEqual(rows["SI"]["pitches"], 2)
        self.assertEqual(rows["SI"]["pitch_pct"], 25)
        self.assertEqual(rows["SI"]["whiff_pct"], 50)
        self.assertEqual(rows["BR"]["pitches"], 2)
        self.assertIsNone(rows["BR"]["whiff_pct"])
        self.assertEqual(rows["OFF"]["whiff_pct"], 100)
        self.assertEqual(sum(r["pitch_pct"] for r in rows.values()), 87.5)

    def test_always_four_rows_missing_metrics_are_blank_and_identity_is_exact(self):
        data = build_splits(self.pitches()).to_pandas().set_index(
            ["batter_mlbid", "season", "level_id", "pitch_group"]
        )
        values = split_values(data, (1, 2026, 1))
        self.assertEqual([r[0] for r in values], ["Fastball", "Sinker", "Breaking", "Offspeed"])
        self.assertEqual(values[2][1 + list(COLUMNS).index("whiff_pct")], "—")
        self.assertIsNone(split_values(data, (1, 2025, 1)))
        self.assertEqual(split_values(data, (1, 2026, 11))[1][1:5], ["0", "0.0", "0", "0"])

    @patch("app.official_hitting._component")
    @patch("app.official_hitting.load_official_hitting", return_value=pd.DataFrame())
    @patch("app.official_hitting.load_hitter_pitch_splits")
    def test_split_section_survives_missing_official_totals_and_excludes_postseason(self, load, official, component):
        load.return_value = build_splits(self.pitches()).to_pandas().set_index(
            ["batter_mlbid", "season", "level_id", "pitch_group"]
        )
        full = pd.DataFrame({"Player ID": [1, 1], "Season": [2026, 2026],
                             "__level": [1, 1], "__official_level": [11, 1],
                             "__official_game_type": ["Regular Season", "Postseason"]})
        render_official_table(pd.DataFrame({"Name": ["Test", "Test"]}), full, "splits")
        data = component.call_args.kwargs["data"]
        self.assertEqual(data["pitch_splits"][0][0][1], "1")
        self.assertIsNone(data["pitch_splits"][1])
        self.assertEqual(data["details"], [None, None])
        self.assertEqual(data["split_columns"][:5], ["Pitch Type", "#", "%", "BBE", "HR"])

    def test_home_runs_are_counted_in_the_correct_regular_season_pitch_group(self):
        pitch = self.pitches().with_columns(
            pl.when(pl.col('pi_pitch_type').is_in(['SI', 'FT'])).then(pl.lit('HR'))
            .otherwise(pl.col('pitch_outcome')).alias('pitch_outcome'))
        for scope in ['hitter', 'team', 'league']:
            result = build_splits(pitch, scope).filter(pl.col('level_id') == 1)
            self.assertEqual(result['HR'].sum(), 2)
            self.assertEqual(result.filter(pl.col('pitch_group') == 'SI')['HR'].sum(), 2)


if __name__ == "__main__":
    unittest.main()

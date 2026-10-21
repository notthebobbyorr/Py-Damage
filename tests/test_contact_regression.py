import unittest
import warnings

import polars as pl

from pipeline.apply_regression_from_agg import apply_contact_from_agg
from pipeline.contact_stability import contact_players, contact_constants


class ContactRegressionTests(unittest.TestCase):
    def setUp(self):
        self.constants = pl.DataFrame({"dataset": ["hitters"], "stat": ["contact_vs_avg"],
            "season": [2026], "level_id": [1], "mu": [3.47], "K": [100.]})

    def test_coe_shrinks_toward_league_mean_without_changing_its_reference_point(self):
        source = pl.DataFrame({"season": [2026]*4, "level_id": [1]*4,
            "contact_vs_avg": [10.7, -15., 10.7, 3.47], "whiff_rate_den": [100, 100, 900, 100],
            "PA": [9999]*4, "pred_whiff_loc_mean": [0.01, 0.99, 0.01, 0.99]})
        out = apply_contact_from_agg(source, self.constants, ["season", "level_id"])
        self.assertEqual(out["contact_vs_avg_raw"].to_list(), source["contact_vs_avg"].to_list())
        self.assertAlmostEqual(out["contact_vs_avg_reg"][0], (10.7 + 3.47)/2)
        self.assertAlmostEqual(out["contact_vs_avg_reg"][1], (-15. + 3.47)/2)
        self.assertGreater(out["contact_vs_avg_reg"][2], out["contact_vs_avg_reg"][0])
        self.assertAlmostEqual(out["contact_vs_avg_reg"][3], 3.47)

    def test_missing_prior_is_unavailable_instead_of_silently_unregressed(self):
        source = pl.DataFrame({"season": [2027], "level_id": [1],
                               "contact_vs_avg": [10.7], "whiff_rate_den": [100]})
        with self.assertWarnsRegex(RuntimeWarning, "insufficient priors"):
            out = apply_contact_from_agg(source, self.constants, ["season", "level_id"])
        self.assertIsNone(out["contact_vs_avg_reg"][0])

    def test_zero_swings_do_not_create_an_estimate(self):
        source = pl.DataFrame({"season": [2026], "level_id": [1],
                               "contact_vs_avg": [None], "whiff_rate_den": [0]},
                              schema_overrides={"contact_vs_avg": pl.Float64})
        with warnings.catch_warnings(record=True) as caught:
            self.assertEqual(apply_contact_from_agg(source, self.constants, ["season", "level_id"]).height, 0)
            self.assertEqual(len(caught), 0)

    def test_priors_use_regular_recorded_swings_and_percentage_point_residuals(self):
        source = pl.DataFrame({"batter_mlbid": [1]*6, "season": [2026]*6, "level_id": [1]*6,
            "game_type": ["R", "R", "R", "S", "R", "R"], "swing": [1, 1, 0, 1, 1, 1],
            "pred_whiff_loc": [0.2, 0.4, 0.9, 0.9, None, 0.9], "whiff": [0, 1, 0, 0, 0, None]})
        player = contact_players(source.lazy()).row(0, named=True)
        self.assertEqual(player["n"], 2)
        self.assertAlmostEqual(player["stat"], -20.)
        self.assertAlmostEqual(player["var"], 3200.)

    def test_league_prior_is_weighted_by_swings_including_small_samples(self):
        player = pl.DataFrame({"season": [2026]*3, "level_id": [1]*3,
            "stat": [10., -10., 30.], "var": [200., 200., 200.], "n": [100, 200, 1]})
        row = contact_constants(player, min_players=2).row(0, named=True)
        self.assertAlmostEqual(row["mu"], (1000-2000+30)/301)
        self.assertGreater(row["K"], 0)


if __name__ == "__main__":
    unittest.main()

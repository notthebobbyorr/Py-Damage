import unittest
import polars as pl

from pipeline.build_combined_splits import SOURCE_COLUMNS, EXTRA_SOURCE, add_dimensions, build_intersection, dimension_sets
from pipeline.data_aggregate import _add_game_type_group, _tag_pitch
from pipeline.backfill_split_workloads import augment, workload


class CombinedSplitTests(unittest.TestCase):
    def test_team_intersections_pool_counts_and_respect_pitch_group_denominators(self):
        pitch=add_dimensions(self.pitches(),True)
        result=build_intersection(pitch,["split_hand","split_pitch_group"],"team_hitter",pl.DataFrame())
        fastball=result.filter((pl.col("split_hand")=="L")&(pl.col("split_pitch_group")=="Fastball"))
        self.assertEqual(fastball["pitches"].item(),2)
        self.assertEqual(fastball["pitch_pct"].item(),50)
        self.assertEqual(result["pitches"].sum(),6)
        self.assertEqual(result["hitting_code"].unique().to_list(),["BOS"])
        pitched=build_intersection(add_dimensions(self.pitches(),False),["split_hand","split_half"],"team_pitcher",pl.DataFrame())
        self.assertEqual(pitched["pitches"].sum(),6)
        self.assertEqual(pitched["bbe"].sum(),2)
        self.assertEqual(pitched["GS"].sum(),0)

    def pitches(self):
        base = dict.fromkeys(SOURCE_COLUMNS+EXTRA_SOURCE, 0.0)
        base.update(batter_mlbid=1,pitcher_mlbid=2,season=2026,level_id=1,game_type="R",
            hitter_name="Test Hitter",name="Test Pitcher",batter_hand="R",pitcher_hand="R",pitcher_role="RP",
            pi_pitch_group="FA",pi_pitch_type="FA",pi_pitch_sub_type="FA",pitch_group="FA",pitch_tag="FA",
            pitching_code="NYY",hitting_code="BOS",home_team="NYY",away_team="BOS",half_inning="top",
            throws="R",stands="L",pitch_outcome="S",primary_tag="FA",game_date="2026-04-01",
            is_inzone_pi=True,is_in_play=False,swing=1,whiff=1,exit_velo=None,launch_angle=None,
            bat_speed=None,swing_length=None,attack_angle=None,attack_direction=None,swing_path_tilt=None,
            intercept_ball_minus_batter_pos_x_inches=None,intercept_ball_minus_batter_pos_y_inches=None)
        rows=[]
        for i,(date,hand,group,ptype,swing,bbe) in enumerate([
                ("2026-04-01","L","FA","FA",1,1),("2026-04-01","L","SL","SL",0,0),
                ("2026-04-01","R","FA","FA",1,0),("2026-08-01","L","FA","FA",1,0),
                ("2026-08-01","L","FA","SI",0,1),("2026-08-01","R","CH","CH",0,0)]):
            rows.append(dict(base,game_date=date,stands=hand,throws=hand,pi_pitch_group=group,pi_pitch_type=ptype,
                             swing=swing,bbe=bbe,game_pk=i,pa_id=i))
        df=pl.DataFrame(rows).with_columns(pl.col("season","level_id").cast(pl.Int64),
            pl.col("exit_velo","launch_angle","bat_speed","swing_length","attack_angle","attack_direction",
                   "swing_path_tilt","intercept_ball_minus_batter_pos_x_inches","intercept_ball_minus_batter_pos_y_inches").cast(pl.Float64))
        return _tag_pitch(_add_game_type_group(df))

    def test_pitch_usage_and_swing_use_the_intersection_denominator(self):
        pitch=add_dimensions(self.pitches(),False)
        result=build_intersection(pitch,["split_hand","split_half"],"pitch",pl.DataFrame())
        rows=result.filter((pl.col("split_hand")=="L") & (pl.col("split_half")=="1st Half"))
        self.assertEqual(rows["pitches"].sum(),2)
        fa=rows.filter(pl.col("pitch_tag")=="FA").row(0,named=True)
        self.assertEqual(fa["pct"],50)
        self.assertEqual(fa["Swing_pct"],100)
        self.assertEqual(fa["bbe"],1)
        self.assertEqual(fa["TBF"],2)  # Workload describes the slice, not just the displayed pitch.
        self.assertEqual(result["pitches"].sum(),6)  # Missing grade calibration must not remove rows.

    def test_hitter_groups_keep_handedness_and_sinkers_separate(self):
        pitch=add_dimensions(self.pitches(),True)
        result=build_intersection(pitch,["split_hand","split_pitch_group"],"hitter",pl.DataFrame())
        left=result.filter(pl.col("split_hand")=="L")
        fa=left.filter(pl.col("split_pitch_group")=="Fastball").row(0,named=True)
        self.assertEqual(fa["pitches"],2)
        self.assertEqual(fa["pitch_pct"],50)
        self.assertEqual(fa["PA"],2)
        self.assertEqual(left.filter(pl.col("split_pitch_group")=="Sinker")["pitches"].item(),1)

    def test_counts_do_not_duplicate_rows_or_replace_established_metrics(self):
        pitch=self.pitches()
        counts=workload(pitch,["pitcher_mlbid"],True)
        original=pl.DataFrame({"pitcher_mlbid":[2.0],"pitches":[99],"SwStr":[7.0]})
        merged=augment(original,counts,["pitcher_mlbid"])
        self.assertEqual(merged.height,1)
        self.assertEqual(merged["pitches"].item(),99)
        self.assertEqual(merged["bbe"].item(),2)
        self.assertEqual(merged["Swing_pct"].item(),50)
        self.assertEqual(merged["GS"].item(),0)

    def test_unrelated_league_ids_do_not_overflow_legacy_join_types(self):
        frame=pl.DataFrame({"level_id":[1]},schema={"level_id":pl.Int8})
        counts=pl.DataFrame({"level_id":[1,586],"bbe":[3,7]})
        result=augment(frame,counts,["level_id"])
        self.assertEqual(result["bbe"].to_list(),[3])

    def test_time_dimensions_are_not_crossed_redundantly(self):
        self.assertFalse(any({"split_half","split_month"}.issubset(d) for d in dimension_sets(True)))

    def test_legacy_pitch_splits_keep_swing_and_slice_workload_on_refresh(self):
        from pipeline.data_aggregate import build_pitching_splits
        calibration=pl.DataFrame({"season":[2026]*4,"pitch_tag":["FA","SI","SL","CH"],
            "stuff_p01":[0.0]*4,"stuff_p99":[1.0]*4})
        _,table=build_pitching_splits(self.pitches(),calibration)
        left=table.filter((pl.col("split_type")=="vs L/R") & (pl.col("split")=="vs L"))
        self.assertEqual(left["pitches"].sum(),4)
        self.assertEqual(left["TBF"].unique().to_list(),[4])
        self.assertEqual(left["GS"].unique().to_list(),[0])
        self.assertEqual(left["bbe"].sum(),2)
        self.assertEqual(left.filter(pl.col("pitch_tag")=="FA")["Swing_pct"].item(),100)


if __name__=="__main__":
    unittest.main()

import unittest
import pandas as pd
import polars as pl
from app.plot_variables import plot_values, default_plot_type, is_month_variable
from pipeline.combined_execution import grade_slices


class PlotVariableTests(unittest.TestCase):
    def test_variable_only_labels_work_without_player_metadata(self):
        from app.viz import _build_point_labels
        frame=pd.DataFrame({"Velo":[95],"Usage (%)":[30]})
        self.assertEqual(_build_point_labels(frame,False,["Velo","Usage (%)"]).iloc[0],"95 | 30")

    def test_month_point_labels_do_not_duplicate_the_split(self):
        from app.viz import _build_point_labels
        frame=pd.DataFrame({"Name":["Pitcher"],"Pitch Type":["FA"],"Split":["May"]})
        label=_build_point_labels(frame,False,["Name","Pitch Type","Split","split"]).iloc[0]
        self.assertEqual(label,"Pitcher | FA | May")

    def test_lines_sort_months_and_never_connect_different_players(self):
        import plotly.express as px
        from app.plot_variables import add_plot_lines, MONTH_ORDER
        frame=pd.DataFrame({"Name":["One","One","Two","Two"],"Split":["May","March/April","May","March/April"],
                            "Pitch Type":["FA"]*4,"Usage":[10,20,30,40]})
        fig=px.scatter(frame,x="Split",y="Usage",color="Pitch Type")
        add_plot_lines(fig,frame,"Split","Usage","Pitch Type",True,MONTH_ORDER)
        lines=[trace for trace in fig.data if trace.mode=="lines"]
        self.assertEqual(len(lines),2)
        self.assertEqual(list(lines[0].x),["March/April","May"])
        self.assertEqual(list(lines[0].y),[20,10])

    def test_legacy_month_split_defaults_to_time_and_retains_season(self):
        values=pd.Series(["September/October","March/April","May"])
        self.assertTrue(is_month_variable(values,"Split"))
        self.assertEqual(default_plot_type(values,"Split"),"Time")
        dates=plot_values(values,"Time","Split",pd.Series([2026]*3))
        self.assertEqual(dates.dt.month.tolist(),[10,4,5])
        self.assertEqual(dates.dt.year.tolist(),[2026]*3)

    def test_time_rejects_pitch_categories_and_numeric_epoch_guesses(self):
        for series in [pd.Series(["FA","SI"]),pd.Series([40,50])]:
            with self.assertRaises(ValueError):plot_values(series,"Time","Split")

    def test_numeric_factors_remain_distinct_categories(self):
        self.assertEqual(plot_values(pd.Series([4,5]),"Discrete / factor","Month").tolist(),["4","5"])

    def test_months_have_real_calendar_coordinates(self):
        self.assertEqual(plot_values(pd.Series(["May","March/April"]),"Continuous","Month").tolist(),[5,3.5])

    def test_pitch_groups_cannot_be_arbitrarily_encoded_as_continuous(self):
        with self.assertRaises(ValueError):plot_values(pd.Series(["Fastball","Sinker"]),"Continuous","Pitch Group")

    def test_execution_grade_is_calibrated_from_the_actual_slice(self):
        rows=[]
        for half,pred,n in [("1st Half",0.25,5),("2nd Half",0.75,5),("Short",0.5,4)]:
            rows += [dict(pitcher_mlbid=1,pitcher_hand="R",season=2026,level_id=1,game_type_group="Regular Season",
                split_hand="L",split_half=half,pitch_tag="FA",pred_v13=pred)]*n
        cal=pl.DataFrame({"season":[2026],"pitch_tag":["FA"],"p1":[0.0],"p99":[1.0]})
        grades=grade_slices(pl.DataFrame(rows),["split_hand","split_half"],cal)
        values={r['split_half']:r['grade_v13'] for r in grades.to_dicts()}
        self.assertEqual(values,{"1st Half":65.0,"2nd Half":35.0,"Short":None})

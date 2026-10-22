import unittest
from unittest.mock import patch
from contextlib import nullcontext
import pandas as pd
import polars as pl
from tempfile import TemporaryDirectory
from pathlib import Path

from app.filters import apply_maximum_filters, split_workload_controls, pitcher_workload_filter, maximum_workload_controls, apply_workload_maximum


class WorkloadFilterTests(unittest.TestCase):
    def test_blank_maximum_and_clear_restore_the_uncapped_sample(self):
        frame=pd.DataFrame({"pitches":[100,200,300],"GS":[0,1,0]})
        state={}
        with patch("app.filters.st.session_state",state):
            self.assertEqual(len(apply_workload_maximum(frame,"bounds")),3)
            state.update(bounds_limit_column="GS",bounds_maximum=0.0)
            self.assertEqual(apply_workload_maximum(frame,"bounds").index.tolist(),[0,2])
            with patch("app.filters.st.selectbox",return_value="GS"),patch("app.filters.st.number_input",return_value=0.0),patch("app.filters.st.button") as button:
                maximum_workload_controls({"# (pitches)":"pitches","GS":"GS"},"bounds")
                button.call_args.kwargs["on_click"]()
            self.assertIsNone(state["bounds_limit_by"])
            self.assertIsNone(state["bounds_maximum"])
            self.assertEqual(len(apply_workload_maximum(frame,"bounds")),3)

    def test_polars_categorical_parquet_loads_for_the_app(self):
        from app.data_loader import _read_parquet
        with TemporaryDirectory(dir=Path.cwd()) as folder:
            path=Path(folder)/"categories.parquet"
            pl.DataFrame({"name":["First","Second"],"bbe":[0,2]}).with_columns(pl.col("name").cast(pl.Categorical)).write_parquet(path)
            frame=_read_parquet(path)
            self.assertEqual(frame.name.tolist(),["First","Second"])
            self.assertEqual(frame.bbe.tolist(),[0,2])
    @patch("app.filters.st.number_input",return_value=1.0)
    @patch("app.filters.st.selectbox",return_value="BBE")
    def test_bbe_is_an_independent_pitching_minimum(self,select,number):
        df=pd.DataFrame({"pitches":[200,200],"bbe":[0,2],"TBF":[50,50],"GS":[0,1],"IP":[10,10]})
        col,minimum=split_workload_controls(df,"test",pitching=True)
        self.assertIn("BBE",select.call_args.args[1])
        self.assertEqual(pitcher_workload_filter(df,col,minimum).index.tolist(),[1])

    @patch("app.filters.st.number_input",return_value=100.0)
    @patch("app.filters.st.selectbox",return_value="# (pitches)")
    def test_splits_default_to_one_hundred_pitches(self,select,number):
        split_workload_controls(pd.DataFrame({"pitches":[1],"bbe":[0]}),"test")
        self.assertEqual(number.call_args_list[0].kwargs["value"],100.0)
        self.assertIsNone(number.call_args_list[1].kwargs["value"])
        self.assertIsNone(select.call_args_list[1].kwargs["index"])

    @patch("app.filters.st.expander",return_value=nullcontext())
    @patch("app.filters.st.multiselect",return_value=["GS","BBE"])
    @patch("app.filters.st.number_input",side_effect=[0.0,20.0])
    def test_maximum_zero_starts_and_bbe_caps_are_inclusive(self,number,select,expander):
        df=pd.DataFrame({"GS":[0,1,0],"BBE":[20,5,21],"#":[200,300,400]},index=[9,3,7])
        self.assertEqual(apply_maximum_filters(df,"test").index.tolist(),[9])

    @patch("app.filters.st.expander",return_value=nullcontext())
    @patch("app.filters.st.multiselect",return_value=["Season GS (regular season)"])
    @patch("app.filters.st.number_input",return_value=0.0)
    @patch("app.data_loader.load_csv")
    def test_hidden_season_workload_respects_identity_level_and_game_type(self,load,number,select,expander):
        load.return_value=pd.DataFrame({"pitcher_mlbid":[1,1,2,2],"season":[2026]*4,"level_id":[1,11,1,1],
            "game_type_group":["Regular Season","Regular Season","Regular Season","Postseason"],"GS":[0,5,3,0],"bbe":[10,20,30,40]})
        display=pd.DataFrame({"Player ID":[2,1,1],"Season":[2026]*3,"__level":[1,11,1],"#":[200]*3},index=[8,7,6])
        result=apply_maximum_filters(display,"test","pitcher")
        self.assertEqual(result.index.tolist(),[6])

    def test_pitch_date_range_swing_rate_uses_summed_counts_in_both_modes(self):
        from app.aggregates import PITCH_TYPE_SPEC, build_span_table
        games=pd.DataFrame({"pitcher_mlbid":[1,1],"pitcher_name":["Test","Test"],"pitch_tag":["FA","FA"],
            "level_id":[1,1],"game_pk":[1,2],"game_date":["2026-04-01","2026-04-02"],
            "pitches":[10,30],"swings":[10,0],"bbe":[2,3],"pitching_code":["NYY","NYY"]})
        for mode in ["Rates","Raw counts"]:
            table=build_span_table(games,PITCH_TYPE_SPEC,mode,"Date","TOTAL")
            self.assertEqual(table.iloc[0]["Swing (%)"],25)
            self.assertEqual(table.iloc[1]["Swing (%)"],0)


if __name__=="__main__":
    unittest.main()

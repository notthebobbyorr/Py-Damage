import unittest
import pandas as pd
from streamlit.testing.v1 import AppTest
from app.comps import prepare_comparison_metrics, translate_fastball_vaa

class PresetTests(unittest.TestCase):
    def test_rates_preserve_missing_denominators(self):
        frame=pd.DataFrame(dict(HR=[2,2],SB=[3,3],PA=[100,0]))
        result=prepare_comparison_metrics(frame,'hitter',True)
        self.assertEqual(result.HR_rate.iloc[0],2)
        self.assertEqual(result.SB_rate.iloc[0],3)
        self.assertTrue(pd.isna(result.HR_rate.iloc[1]))
        self.assertTrue(result.HR_rate.equals(result.HR_rate_mlb_eq))

    def test_presets_and_custom_edits_all_pages(self):
        for kind in ['hitter','pitcher','pitch']:
            for equivalent in ([False,True] if kind!='pitch' else [False]):
                script=f"""
import pandas as pd
import streamlit as st
from app.comps import PRESETS,comparison_columns
kind='{kind}'
suffix={'_mlb_eq' if equivalent else ''!r}
cols=list(dict.fromkeys(c+suffix for v in PRESETS[kind].values() for c in v.split()))
frame=pd.DataFrame({{c:[1.,2.] for c in cols}})
selected,_=comparison_columns(kind,frame,{{}},cols[:2],'test',equivalent={equivalent})
st.session_state['selected']=selected
"""
                app=AppTest.from_string(script).run()
                self.assertFalse(app.warning)
                self.assertEqual(len(app.session_state['selected']), 2)
                app.run()
                self.assertFalse(app.warning)
                from app.comps import PRESETS
                for name, columns in PRESETS[kind].items():
                    app.selectbox(key='test_preset').select(name).run()
                    self.assertFalse(app.exception)
                    self.assertFalse(app.warning)
                    suffix='_mlb_eq' if equivalent else ''
                    self.assertEqual(app.session_state['selected'],[c+suffix for c in columns.split()])
                app.multiselect(key='test').set_value([]).run()
                self.assertEqual(app.selectbox(key='test_preset').value,'Custom')
                self.assertFalse(app.warning)

    def test_vaa_only_changes_minor_fastballs(self):
        pitchers=pd.DataFrame(dict(pitcher_mlbid=[1,2,3,4],season=[2026]*4,level_id=[1,1,14,14],fastball_vaa_reg=[-6.,-4.,-5.,-3.]))
        coeff=pd.DataFrame(dict(metric=['fastball_vaa_reg'],src_level=[14],dst_level=[1],a=[0.],b=[1.]))
        pitches=pd.DataFrame(dict(season=[2026]*3,level_id=[14,14,1],pitch_group=['FA','BR','FA'],vaa=[-4.]*3))
        result=translate_fastball_vaa(pitches,pitchers,coeff)
        self.assertEqual(result.vaa.tolist(),[-5.,-4.,-4.])

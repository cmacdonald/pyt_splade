import unittest
import pandas as pd
import tempfile
class TestBasic(unittest.TestCase):

    def setUp(self):
        import pyterrier as pt
        if not pt.started():
            pt.init()

    def test_transformer(self):
        import pyt_splade
        t = pyt_splade.splade()
        df = (t >> pyt_splade.toks2doc()).transform_iter([{'docno' : 'd1', 'text' : 'hello there'}])
        self.assertTrue('there there' in df.iloc[0].text)
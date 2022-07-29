import unittest
import pandas as pd
import tempfile
class TestBasic(unittest.TestCase):

    def setUp(self):
        import pyterrier as pt
        if not pt.started():
            pt.init()
        import pyt_splade
        self.t = pyt_splade.splade()

    def test_transformer1(self):
        import pyt_splade
        df = (self.t >> pyt_splade.toks2doc()).transform_iter([{'docno' : 'd1', 'text' : 'hello there'}])
        self.assertTrue('there there' in df.iloc[0].text)
        df = self.t.transform_iter([{'docno' : 'd1', 'text' : 'hello there'}, {'docno' : 'd1', 'text' : 'hello hello hello hello hello there'}])
        
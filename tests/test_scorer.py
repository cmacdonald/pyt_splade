import unittest
import pandas as pd
import tempfile
class TestBasic(unittest.TestCase):

    def setUp(self):
        import pyterrier as pt
        if not pt.started():
            pt.init()
        import pyt_splade
        self.factory = pyt_splade.SpladeFactory(device='cpu')

    def test_scorer(self):
        df = self.factory.scorer()(pd.DataFrame([
          {'qid': '0', 'query': 'chemical reactions', 'docno' : 'd1', 'text' : 'hello there'},
          {'qid': '0', 'query': 'chemical reactions', 'docno' : 'd2', 'text' : 'chemistry society'},
          {'qid': '1', 'query': 'hello', 'docno' : 'd1', 'text' : 'hello there'},
        ]))
        print(df)

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
        self.assertAlmostEqual(0., df['score'][0])
        self.assertAlmostEqual(11.133593, df['score'][1], places=5)
        self.assertAlmostEqual(17.566324, df['score'][2], places=5)
        self.assertEqual('0', df['qid'][0])
        self.assertEqual('0', df['qid'][1])
        self.assertEqual('1', df['qid'][2])
        self.assertEqual('d1', df['docno'][0])
        self.assertEqual('d2', df['docno'][1])
        self.assertEqual('d1', df['docno'][2])
        self.assertEqual(1, df['rank'][0])
        self.assertEqual(0, df['rank'][1])
        self.assertEqual(0, df['rank'][2])

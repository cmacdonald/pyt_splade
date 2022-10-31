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

    def test_transformer_indexing(self):
        import pyt_splade
        df = (self.factory.indexing() >> pyt_splade.toks2doc()).transform_iter([{'docno' : 'd1', 'text' : 'hello there'}])
        self.assertTrue('there there' in df.iloc[0].text)
        df = self.factory.indexing().transform_iter([
            {'docno' : 'd1', 'text' : 'hello there'}, 
            {'docno' : 'd1', 'text' : ''}, #empty 
            {'docno' : 'd1', 'text' : 'hello hello hello hello hello there'}])

    def test_transformer_querying(self):
            q = self.factory.query()
            df = q.transform_iter([{'qid' : 'q1', 'query' : 'chemical reactions'}])
            print(df.iloc[0].query)
            self.assertTrue('#combine' in df.iloc[0].query)

    def test_transformer_empty_query(self):
        q = self.factory.query()
        res = q(pd.DataFrame([], columns=['qid', 'query']))
        self.assertEqual(['qid', 'query_0', 'query'], list(res.columns))

    def test_transformer_empty_doc(self):
        d = self.factory.indexing()
        res = d(pd.DataFrame([], columns=['docno', 'text']))
        self.assertEqual(['docno', 'text', 'toks'], list(res.columns))

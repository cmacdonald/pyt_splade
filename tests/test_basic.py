import unittest
import pandas as pd
from unittest.mock import MagicMock
import pyt_splade

class TestBasic(unittest.TestCase):

    def setUp(self):
        self.factory = pyt_splade.Splade(device='cpu')

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

    def test_model_output_one_dim_non_zero_rep(self):
        import torch
        one_dim_non_zero = torch.zeros(1, self.factory.model.output_dim)
        one_dim_non_zero[0][0] = 1.
        mock_return = {
            "d_rep": one_dim_non_zero,
            "q_rep": one_dim_non_zero,
        }
        factory = pyt_splade.SpladeFactory(device='cpu')
        mock_model = MagicMock(return_value=mock_return)
        factory.model = mock_model

        res = factory.indexing()(
            [{'docno' : 'd1', 'text' : 'hello there'}]
        )
        self.assertEqual(['docno', 'text', 'toks'], list(res.columns))

        res = factory.query()(
            [{'qid' : 'd1', 'query' : 'chemical reactions'}]
        )
        self.assertEqual(['qid', 'query_0', 'query'], list(res.columns))

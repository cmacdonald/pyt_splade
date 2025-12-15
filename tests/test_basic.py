import unittest
import pandas as pd
from unittest.mock import MagicMock
import pyt_splade
import pyterrier as pt

class TestBasic(unittest.TestCase):

    def setUp(self):
        self.splade = pyt_splade.Splade(device='cpu')

    def test_transformer_indexing(self):
        df = (self.splade.doc_encoder() >> pyt_splade.toks2doc())(pd.DataFrame([{'docno' : 'd1', 'text' : 'hello there'}]))
        self.assertTrue('there there' in df.iloc[0].text)
        df = self.splade.doc_encoder()([
            {'docno' : 'd1', 'text' : 'hello there'}, 
            {'docno' : 'd1', 'text' : ''}, #empty 
            {'docno' : 'd1', 'text' : 'hello hello hello hello hello there'}])

    def test_transformer_querying(self):
        q = self.splade.query_encoder()
        df = q(pd.DataFrame([{'qid' : 'q1', 'query' : 'chemical reactions'}]))
        self.assertTrue('query_toks' in df.columns)

    def test_transformer_empty_query(self):
        q = self.splade.query_encoder()
        self.assertEqual([["qid", "query"]], pt.inspect.transformer_inputs(q))
        self.assertEqual(["qid", "query", "query_toks"], pt.inspect.transformer_outputs(q, ["qid", "query"]))
        res = q(pd.DataFrame([], columns=['qid', 'query']))
        self.assertEqual(['qid', 'query', 'query_toks'], list(res.columns))

    def test_transformer_empty_doc(self):
        d = self.splade.doc_encoder()
        self.assertEqual([["docno", "text"]], pt.inspect.transformer_inputs(d))
        self.assertEqual(["docno", "text", "toks"], pt.inspect.transformer_outputs(d, ["docno", "text"]))
        res = d(pd.DataFrame([], columns=['docno', 'text']))
        self.assertEqual(['docno', 'text', 'toks'], list(res.columns))

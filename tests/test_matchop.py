import unittest
import pandas as pd
import tempfile
class TestMatchop(unittest.TestCase):

    def setUp(self):
        import pyterrier as pt
        if not pt.started():
            pt.init()
        import pyt_splade
        self.factory = pyt_splade.SpladeFactory(gpu=False)

    def test_it(self):
        from pyt_splade import _matchop
        self.assertEqual(_matchop('a', 1), 'a')
        self.assertEqual(_matchop('a', 1.1), '#combine:0=1.1(a)')
        self.assertEqual(_matchop('##a', 1.1), '#combine:0=1.1(#base64(IyNh))') 
                

import unittest
from pyt_splade._utils import _matchop

class TestMatchop(unittest.TestCase):

    def test_it(self):
        
        self.assertEqual(_matchop('a', 1), 'a')
        self.assertEqual(_matchop('a', 1.1), '#combine:0=1.1(a)')
        self.assertEqual(_matchop('##a', 1.1), '#combine:0=1.1(#base64(IyNh))') 

        self.assertTrue("#base64" in _matchop('"', 1))
                

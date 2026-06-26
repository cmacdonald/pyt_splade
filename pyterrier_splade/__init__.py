__version__ = '0.1.0'

from pyterrier_splade._model import Splade
from pyterrier_splade._encoder import SpladeEncoder
from pyterrier_splade._scorer import SpladeScorer
from pyterrier_splade._utils import Toks2Doc

SpladeFactory = Splade # backward compatible name
toks2doc = Toks2Doc # backward compatible name

__all__ = ['Splade', 'SpladeEncoder', 'SpladeScorer', 'SpladeFactory', 'Toks2Doc', 'toks2doc']

__version__ = '0.0.2'

from pyt_splade._model import Splade
from pyt_splade._encoder import SpladeEncoder
from pyt_splade._scorer import SpladeScorer
from pyt_splade._utils import Toks2Doc

SpladeFactory = Splade # backward compatible name
toks2doc = Toks2Doc # backward compatible name

__all__ = ['Splade', 'SpladeEncoder', 'SpladeScorer', 'SpladeFactory', 'Toks2Doc', 'toks2doc']

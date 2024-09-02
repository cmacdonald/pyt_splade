import pandas as pd
import pyterrier as pt


class Toks2Doc(pt.Transformer):
    """Converts a toks field into a text field, by scaling the weights by ``mult`` and repeating them."""
    def __init__(self, mult: float = 100.):
        """Initializes the transformer.

        Args:
            mult: the multiplier to apply to the term frequencies
        """
        self.mult = mult

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        """Converts the toks field into a text field."""
        res = inp.assign(text=inp['toks'].apply(self._dict_tf2text))
        res.drop(columns=['toks'], inplace=True)
        return res

    def _dict_tf2text(self, tfdict):
        rtr = ""
        for t in tfdict:
            for i in range(int(self.mult * tfdict[t])):
                rtr += t + " "
        return rtr

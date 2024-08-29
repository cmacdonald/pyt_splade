import base64
import string
import pandas as pd
import pyterrier_alpha as pta
import pyterrier as pt


class MatchOp(pt.Transformer):
    """Converts a query_toks field into a query field, using the MatchOp syntax."""

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Converts the query_toks field into a query field."""
        pta.validate.query_frame(df, ['query_toks'])
        rtr = pt.model.push_queries(df)
        rtr = rtr.assign(query=df.query_toks.apply(lambda toks: ' '.join(_matchop(k, v) for k, v in toks.items())))
        rtr = rtr.drop(columns=['query_toks'])
        return rtr


def _matchop(t, w):
    """Converts a term and its weight into MatchOp syntax."""
    if not all(a in string.ascii_letters + string.digits for a in t):
        encoded = base64.b64encode(t.encode('utf-8')).decode("utf-8")
        t = f'#base64({encoded})'
    if w != 1:
        t = f'#combine:0={w}({t})'
    return t


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
        res = inp.assign(toks=inp['toks'].apply(self._dict_tf2text))
        res.drop(columns=['toks'], inplace=True)
        return res

    def _dict_tf2text(self, tfdict):
        rtr = ""
        for t in tfdict:
            for i in range(int(self.mult * tfdict[t])):
                rtr += t + " "
        return rtr

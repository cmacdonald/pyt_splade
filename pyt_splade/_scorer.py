import more_itertools
import pandas as pd
import numpy as np
import pyterrier as pt
import pyterrier_alpha as pta

class SpladeScorer(pt.Transformer):
    """Scores (re-ranks) documents against queries using a SPLADE model."""
    def __init__(self, splade, text_field, batch_size=100, verbose=False):
        """Initializes the SPLADE scorer.

        Args:
            splade: :class:`pyt_splade.Splade` instance
            text_field: the text field to score
            batch_size: the batch size to use when scoring
            verbose: if True, show a progress bar
        """
        self.splade = splade
        self.text_field = text_field
        self.batch_size = batch_size
        self.verbose = verbose

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scores (re-ranks) the documents against the queries in the input DataFrame."""
        pta.validate.results_frame(df, ['query', self.text_field])
        it = df.groupby('query')
        if self.verbose:
            it = pt.tqdm(it, unit='query')
        res = []
        for query, df in it:
            query_enc = self.splade.encode([query], 'q', 'torch')
            scores = []
            for batch in more_itertools.chunked(df[self.text_field], self.batch_size):
                doc_enc = self.splade.encode(batch, 'd', 'torch')
                scores.append((query_enc @ doc_enc.T).flatten().cpu().numpy())
            res.append(df.assign(score=np.concatenate(scores)))
        res = pd.concat(res)
        from pyterrier.model import add_ranks
        res = add_ranks(res)
        return res

from typing import Literal
import more_itertools
import pandas as pd
import pyterrier as pt
import pyterrier_alpha as pta
import pyt_splade


class SpladeEncoder(pt.Transformer):
    """Encodes a text field using a SPLADE model. The output is a dense or sparse representation of the text field."""

    def __init__(
        self,
        splade: pyt_splade.Splade,
        text_field: str,
        out_field: str,
        rep: Literal['q', 'd'],
        sparse: bool = True,
        batch_size: int = 100,
        verbose: bool = False,
        scale: float = 1.,
    ):
        """Initializes the SPLADE encoder.

        Args:
            splade: :class:`pyt_splade.Splade` instance
            text_field: the input text field to encode
            out_field: the output field to store the encoded representation
            rep: 'q' for query, 'd' for document
            sparse: if True, the output will be a dict of term frequencies, otherwise a dense vector
            batch_size: the batch size to use when encoding
            verbose: if True, show a progress bar
            scale: the scale to apply to the term frequencies
        """
        self.splade = splade
        self.text_field = text_field
        self.out_field = out_field
        self.rep = rep
        self.sparse = sparse
        self.batch_size = batch_size
        self.verbose = verbose
        self.scale = scale

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encodes the text field in the input DataFrame."""
        pta.validate.columns(df, includes=[self.text_field])
        it = iter(df[self.text_field])
        if self.verbose:
            it = pt.tqdm(it, total=len(df), unit=self.text_field)
        res = []
        for batch in more_itertools.chunked(it, self.batch_size):
            res.extend(self.splade.encode(batch, self.rep, format='dict' if self.sparse else 'np', scale=self.scale))
        return df.assign(**{self.out_field: res})

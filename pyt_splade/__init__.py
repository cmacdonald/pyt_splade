import base64
import string
import more_itertools
from multiprocessing.sharedctypes import Value
import pyterrier as pt
assert pt.started()
from typing import Union
import torch
import numpy as np
import pandas as pd


class Splade():

    def __init__(
        self,
        model : Union[torch.nn.Module, str] = "naver/splade-cocondenser-ensembledistil",
        tokenizer=None,
        agg='max',
        max_length = 256,
        device=None):

        import torch
        self.max_length = max_length
        self.model = model
        self.tokenizer = tokenizer
        if device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        else:
            self.device = torch.device(device)
        if isinstance(model, str):
            from splade.models.transformer_rep import Splade
            if self.tokenizer is None:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model)
            self.model = Splade(model, agg=agg)
            self.model.eval()
            self.model = self.model.to(self.device)
        else:
            if self.tokenizer is None:
                raise ValueError("you must specify tokenizer if passing a model")

        self.reverse_voc = {v: k for k, v in self.tokenizer.vocab.items()}

    def doc_encoder(self, field='text', batch_size=100, sparse=True, verbose=False) -> pt.Transformer:
        return SpladeEncoder(self, field, 'toks' if sparse else 'doc_vec', 'd', sparse, batch_size, verbose)
    indexing = doc_encoder # backward compatible name

    def query_encoder(self, batch_size=100, sparse=True, verbose=False) -> pt.Transformer:
        return SpladeEncoder(self, 'query', 'query_toks' if sparse else 'query_vec', 'q', sparse, batch_size, verbose)

    def matchop_query_encoder(self, batch_size=100, verbose=False, mult=100) -> pt.Transformer:
        return self.query_encoder(batch_size, sparse=True, verbose=verbose) >> MatchOp(mult=mult)
    query = matchop_query_encoder # backward compatible name

    def scorer(self, field='text', batch_size=100, verbose=False) -> pt.Transformer:
        return SpladeScorer(self, field, batch_size, verbose)

    def encode(self, texts, rep='d', format='dict'):
        rtr = []
        with torch.no_grad():
            reps = self.model(**{rep+'_kwargs': self.tokenizer(
                texts,
                add_special_tokens=True,
                padding="longest",  # pad to max sequence length in batch
                truncation="longest_first",  # truncates to max model length,
                max_length=self.max_length,
                return_attention_mask=True,
                return_tensors="pt",
            ).to(self.device)})[rep+'_rep']
        if format == 'dict':
            reps = reps.cpu()
            for i in range(reps.shape[0]):
                # get the number of non-zero dimensions in the rep:
                col = torch.nonzero(reps[i]).squeeze().tolist()
                # now let's create the bow representation as a dictionary
                weights = reps[i, col].cpu().tolist()
                d = {self.reverse_voc[k] : v for k, v in sorted(zip(col, weights), key=lambda x: (-x[1], x[0]))}
                rtr.append(d)
        elif format == 'np':
            reps = reps.cpu().numpy()
            for i in range(reps.shape[0]):
                rtr.append(reps[i])
        elif format == 'torch':
            rtr = reps
        return rtr


SpladeFactory = Splade # backward compatible name


class SpladeEncoder(pt.Transformer):
    def __init__(self, splade, field, out_field, rep, sparse=True, batch_size=100, verbose=False):
        self.splade = splade
        self.field = field
        self.out_field = out_field
        self.rep = rep
        self.sparse = sparse
        self.batch_size = batch_size
        self.verbose = verbose

    def transform(self, df):
        assert self.field in df.columns
        it = iter(df[self.field])
        if self.verbose:
            it = pt.tqdm(it, total=len(df), unit=self.field)
        res = []
        for batch in more_itertools.chunked(it, self.batch_size):
            res.extend(self.splade.encode(batch, self.rep, format='dict' if self.sparse else 'np'))
        return df.assign(**{self.out_field: res})


class SpladeScorer(pt.Transformer):
    def __init__(self, splade, field, batch_size=100, verbose=False):
        self.splade = splade
        self.field = field
        self.batch_size = batch_size
        self.verbose = verbose

    def transform(self, df):
        assert all(f in df.columns for f in ['query', self.field])
        it = df.groupby('query')
        if self.verbose:
            it = pt.tqdm(it, unit='query')
        res = []
        for query, df in it:
            query_enc = self.splade.encode([query], 'q', 'torch')
            scores = []
            for batch in more_itertools.chunked(df[self.field], self.batch_size):
                doc_enc = self.splade.encode(batch, 'd', 'torch')
                scores.append((query_enc @ doc_enc.T).flatten().cpu().numpy())
            res.append(df.assign(score=np.concatenate(scores)))
        res = pd.concat(res)
        from pyterrier.model import add_ranks
        res = add_ranks(res)
        return res


class MatchOp(pt.Transformer):
    def __init__(self, mult=100):
        self.mult = mult

    def transform(self, df):
        assert 'query_toks' in df.columns
        from pyterrier.model import push_queries
        rtr = push_queries(df)
        rtr = rtr.assign(query=df.query_toks.apply(lambda toks: ' '.join(_matchop(k, v * self.mult) for k, v in toks.items())))
        rtr = rtr.drop(columns=['query_toks'])
        return rtr


def _matchop(t, w):
    if not all(a in string.ascii_letters + string.digits for a in t):
        encoded = base64.b64encode(t.encode('utf-8')).decode("utf-8")
        t = f'#base64({encoded})'
    if w != 1:
        t = f'#combine:0={w}({t})'
    return t



def toks2doc(mult=100):
    
    def _dict_tf2text(tfdict):
        rtr = ""
        for t in tfdict:
            for i in range(int(mult*tfdict[t])):
                rtr += t + " " 
        return rtr

    def _rowtransform(df):
        df = df.copy()
        df["text"] = df['toks'].apply(_dict_tf2text)
        df.drop(columns=['toks'], inplace=True)
        return df
    
    return pt.apply.generic(_rowtransform)
    

from multiprocessing.sharedctypes import Value
import pyterrier as pt
assert pt.started()
from typing import Union
import torch

def splade(
        model : Union[torch.nn.Module, str] = "naver/splade-cocondenser-ensembledistil", 
        tokenizer=None,
        agg='max',
    ) -> pt.Transformer:
    
    import torch
    from transformers import AutoModelForMaskedLM
    if isinstance(model, str):
        from splade.models.transformer_rep import Splade
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model)
        model = Splade(model, agg=agg)
        model.eval()
        
    else:
        if tokenizer is None:
            raise ValueError("you  must specify tokenizer if passing a model")

    import pandas as pd
    reverse_voc = {v: k for k, v in tokenizer.vocab.items()}

    def _transform(df):
        rtr = []
        #TODO batching
        with torch.no_grad():
            for row in df.itertuples():
                # now compute the document representation
                doc_rep = model(d_kwargs=tokenizer(row.text, return_tensors="pt"))["d_rep"].squeeze()  # (sparse) doc rep in voc space, shape (30522,)

                # get the number of non-zero dimensions in the rep:
                col = torch.nonzero(doc_rep).squeeze().cpu().tolist()
                print("number of actual dimensions: ", len(col))

                # now let's inspect the bow representation:
                weights = doc_rep[col].cpu().tolist()
                d = {reverse_voc[k] : v for k, v in zip(col, weights)}

                rtr.append([row.docno, d])
        return pd.DataFrame(rtr, columns=['docno', 'toks'])
    return pt.apply.generic(_transform)

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
    

from multiprocessing.sharedctypes import Value
import pyterrier as pt
assert pt.started()
from typing import Union
import torch

def splade(
        model : Union[torch.nn.Module, str] = "naver/splade-cocondenser-ensembledistil", 
        tokenizer=None,
        agg='max',
        max_length = 256,
        gpu=True
    ) -> pt.Transformer:
    
    import torch
    device = torch.device('cuda') if gpu else torch.device('cpu')
    from transformers import AutoModelForMaskedLM
    if isinstance(model, str):
        from splade.models.transformer_rep import Splade
        if tokenizer is None:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model)
        model = Splade(model, agg=agg)
        model.eval()
        if gpu:
            model.to(device)
        
    else:
        if tokenizer is None:
            raise ValueError("you  must specify tokenizer if passing a model")

    import pandas as pd
    reverse_voc = {v: k for k, v in tokenizer.vocab.items()}

    def _transform(df):
        rtr = []
        with torch.no_grad():
            # now compute the document representation
            doc_reps = model(d_kwargs=tokenizer(
                df.text.tolist(),
                add_special_tokens=True,
                padding="longest",  # pad to max sequence length in batch
                truncation="longest_first",  # truncates to max model length,
                max_length=max_length,
                return_attention_mask=True,
                return_tensors="pt",
                ).to(device))["d_rep"]  # (sparse) doc rep in voc space, shape (docs, 30522,)

            for i in range(doc_reps.shape[0]): #for each doc
                # get the number of non-zero dimensions in the rep:
                col = torch.nonzero(doc_reps[i]).squeeze().cpu().tolist()

                # now let's inspect the bow representation:                
                weights = doc_reps[i,col].cpu().tolist()
                d = {reverse_voc[k] : v for k, v in zip(col, weights)}
                rtr.append([df.iloc[i].docno, d])
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
    

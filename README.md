# pyterrier_splade

An example of a SPLADE indexing and retrieval using PyTerrier transformers. 


# Installation

We use [Naver's SPLADE repository](https://github.com/naver/splade) as a dependency: 

```python
%pip install -q python-terrier
%pip install -q git+https://github.com/naver/splade.git git+https://github.com/cmacdonald/pyt_splade.git
```

# Indexing

Indexing takes place as a pipeline: we apply SPLADE transformation of the documents, which maps raw text into a dictionary of BERT WordPiece tokens and corresponding weights. The underlying indexer, Terrier, is configured to handle arbitrary word counts without further tokenisation (`pretokenised=True`).

The Terrier indexer is configured to index tokens unchanged. 

```python

import pyterrer as pt
pt.init()

import pyt_splade
factory = pyt_splade.SpladeFactory()
indexer = pt.IterDictIndexer('./msmarco_psg', pretokenised=True)

indxr_pipe = factory.indexing() >> indexer
index_ref = indxr_pipe.index(dataset.get_corpus_iter(), batch_size=128)

```

# Retrieval

Similarly, SPLADE encodes the query into BERT WordPieces and corresponding weights.
We apply this as a query encoding transformer. It encodes the query into Terrier's matchop query language, to avoid tokenisation problems.

```python

splade_retr = factory.query() >> pt.BatchRetrieve('./msmarco_psg', wmodel='Tf')

```

# Credits 

Craig Macdonald

SPLADE Overview
==========================================

`pyterrier-splade <https://github.com/cmacdonald/pyt_splade>`__ lets you construct learned sparse indexing and
retrieval pipelines. SPLADE maps queries and documents into sparse bags of weighted BERT WordPiece tokens, combining
the efficiency of sparse retrieval with the effectiveness of learned term weighting and expansion.

A pipeline is built from two main pieces: a **SPLADE model**, which encodes text into weighted tokens, and a
**sparse index** (Terrier or PISA), which stores those tokens and retrieves over them.

SPLADE Model
------------------------------------------

A :class:`~pyterrier_splade.Splade` model is a factory for the transformers used to encode and score text.

.. code-block:: python
   :caption: Loading a SPLADE model

   import pyterrier_splade
   splade = pyterrier_splade.Splade() # :footnote: You can pass any SPLADE model name or path; it defaults to ``naver/splade-cocondenser-ensembledistil``.

A SPLADE model can produce several transformers:

- Encode documents into weighted tokens using its :meth:`~pyterrier_splade.Splade.doc_encoder`.
- Encode queries into weighted tokens using its :meth:`~pyterrier_splade.Splade.query_encoder`.
- Re-rank results by scoring query/document pairs using its :meth:`~pyterrier_splade.Splade.scorer`.

Indexing
------------------------------------------

Indexing takes place as a pipeline: the SPLADE document encoder maps raw text into a dictionary of BERT WordPiece tokens
and corresponding weights, and the underlying sparse indexer stores them. Terrier is configured to index the tokens
unchanged, without further tokenisation or stemming (``pretokenised=True``).

.. code-block:: python
   :caption: Indexing documents with SPLADE

   import pyterrier as pt
   import pyterrier_splade

   splade = pyterrier_splade.Splade()
   indexer = pt.IterDictIndexer('./msmarco_psg', pretokenised=True)

   idx_pipeline = splade.doc_encoder() >> indexer
   idx_pipeline.index(dataset.get_corpus_iter(), batch_size=128)

Retrieval
------------------------------------------

Similarly, SPLADE encodes the query into weighted BERT WordPieces, which are passed to a sparse retriever. Scoring the
SPLADE term weights with ``wmodel='Tf'`` recovers the SPLADE dot-product score.

.. code-block:: python
   :caption: Retrieving with SPLADE

   splade_retr = splade.query_encoder() >> pt.terrier.Retriever('./msmarco_psg', wmodel='Tf')

Scoring
------------------------------------------

SPLADE can also be used as a text scoring (re-ranking) function over the results of an earlier stage.

.. code-block:: python
   :caption: Re-ranking with the SPLADE scorer

   first_stage = ... # e.g., BM25, dense retrieval, etc.
   splade_scorer = first_stage >> dataset.text_loader() >> splade.scorer()

PISA
------------------------------------------

For faster retrieval, you can use the `PISA <https://github.com/terrierteam/pyterrier_pisa>`__ backend in place of Terrier.

.. code-block:: python
   :caption: Indexing and retrieving with SPLADE over a PISA index

   import pyterrier as pt
   import pyterrier_splade
   from pyterrier_pisa import PisaIndex

   splade = pyterrier_splade.Splade()
   dataset = pt.get_dataset('irds:msmarco-passage')
   index = PisaIndex('./msmarco-passage-splade', stemmer='none')

   # indexing
   idx_pipeline = splade.doc_encoder() >> index.toks_indexer()
   idx_pipeline.index(dataset.get_corpus_iter())

   # retrieval
   retr_pipeline = splade.query_encoder() >> index.quantized()

BMP
------------------------------------------

`BMP (Block-Max Pruning) <https://github.com/pisa-engine/BMP>`__ is another fast sparse retrieval backend that works
well with learned sparse models like SPLADE. Install it with ``pip install bmp[pyterrier]``.

.. code-block:: python
   :caption: Indexing and retrieving with SPLADE over a BMP index

   import pyterrier_splade
   from bmp.pyterrier import BmpIndex

   splade = pyterrier_splade.Splade()
   index = BmpIndex('./msmarco-passage-splade.bmp') # :footnote: The ``.bmp`` extension is optional.

   # indexing
   idx_pipeline = splade.doc_encoder() >> index.indexer()
   idx_pipeline.index(dataset.get_corpus_iter())

   # retrieval
   retr_pipeline = splade.query_encoder() >> index.retriever()
   retr_pipeline.search('my query')

.. seealso::
    Step-by-step recipes for these pipelines are available on the :doc:`how-to` page.

SPLADE How-To Guides
============================================================


.. how-to:: How do I index documents with SPLADE?

    .. code-block:: python
        :caption: Indexing documents with SPLADE into a Terrier index

        import pyterrier as pt
        import pyterrier_splade

        splade = pyterrier_splade.Splade() # :footnote: Specify the SPLADE model to use; defaults to ``naver/splade-cocondenser-ensembledistil``.
        indexer = pt.IterDictIndexer('./msmarco_psg', pretokenised=True) # :footnote: ``pretokenised=True`` tells Terrier to index the SPLADE tokens unchanged, without further tokenisation or stemming.
        idx_pipeline = splade.doc_encoder() >> indexer # :footnote: Create an indexing pipeline by chaining the SPLADE document encoder and the indexer.

        idx_pipeline.index(dataset.get_corpus_iter(), batch_size=128) # :footnote: ``get_corpus_iter()`` can be any *iterable* of documents, including generators. This allows you to index collections that are too large to fit in memory at once.


.. how-to:: How do I retrieve with SPLADE?

    .. _pyterrier-splade:how-to:terrier-retrieval:

    This example assumes that you already built a SPLADE index for your collection (see the guide above). For faster
    retrieval, check out :ref:`the PISA guide <pyterrier-splade:how-to:pisa>`.

    .. code-block:: python
        :caption: Retrieving over a SPLADE Terrier index

        import pyterrier as pt
        import pyterrier_splade

        splade = pyterrier_splade.Splade() # :footnote: Specify the model used to create the index.
        retr = splade.query_encoder() >> pt.terrier.Retriever('./msmarco_psg', wmodel='Tf') # :footnote: Create a retrieval pipeline by chaining the SPLADE query encoder with a Terrier retriever. ``wmodel='Tf'`` scores documents by the SPLADE term weights.

        results = retr.search('a single query')
        # or
        results = retr([
            {'qid': '1', 'query': 'multiple queries'},
            {'qid': '2', 'query': 'can be passed as a list of dicts'},
        ])


.. how-to:: How do I re-rank with SPLADE?

    .. code-block:: python
        :caption: Re-ranking initial results with the SPLADE scorer

        import pyterrier as pt
        import pyterrier_splade

        splade = pyterrier_splade.Splade() # :footnote: Specify the model you want to use as a re-ranker.
        first_stage = pt.terrier.Retriever('./msmarco_psg', wmodel='BM25') # :footnote: In this example, we use BM25 over a sparse index for initial retrieval.
        retr = first_stage >> dataset.text_loader() >> splade.scorer() # :footnote: Create a re-ranking pipeline by chaining an initial retriever, a text loader, and the SPLADE scorer. ``text_loader`` loads the document text required by the scorer.

        retr.search('my query')


.. how-to:: How do I retrieve faster with PISA?

    .. _pyterrier-splade:how-to:pisa:

    For faster retrieval, you can use the `PISA <https://github.com/terrierteam/pyterrier_pisa>`__ backend instead of Terrier.

    .. code-block:: python
        :caption: Indexing and retrieving with SPLADE over a PISA index

        import pyterrier as pt
        import pyterrier_splade
        from pyterrier_pisa import PisaIndex

        splade = pyterrier_splade.Splade() # :footnote: Specify the SPLADE model to use.
        index = PisaIndex('./msmarco-passage-splade', stemmer='none') # :footnote: ``stemmer='none'`` keeps the SPLADE tokens unchanged.

        idx_pipeline = splade.doc_encoder() >> index.toks_indexer() # :footnote: Create an indexing pipeline by chaining the SPLADE document encoder and the PISA tokens indexer.
        idx_pipeline.index(dataset.get_corpus_iter())

        retr = splade.query_encoder() >> index.quantized() # :footnote: Create a retrieval pipeline by chaining the SPLADE query encoder with the quantized PISA retriever.
        retr.search('my query')


.. how-to:: How do I retrieve faster with BMP?

    .. _pyterrier-splade:how-to:bmp:

    `BMP (Block-Max Pruning) <https://github.com/pisa-engine/BMP>`__ is another fast sparse retrieval backend. Install it
    with ``pip install bmp[pyterrier]``.

    .. code-block:: python
        :caption: Indexing and retrieving with SPLADE over a BMP index

        import pyterrier as pt
        import pyterrier_splade
        from bmp.pyterrier import BmpIndex

        splade = pyterrier_splade.Splade() # :footnote: Specify the SPLADE model to use.
        index = BmpIndex('./msmarco-passage-splade.bmp') # :footnote: Specify the path to index to; the ``.bmp`` extension is optional.

        idx_pipeline = splade.doc_encoder() >> index.indexer() # :footnote: Create an indexing pipeline by chaining the SPLADE document encoder and the BMP indexer.
        idx_pipeline.index(dataset.get_corpus_iter())

        retr = splade.query_encoder() >> index.retriever() # :footnote: Create a retrieval pipeline by chaining the SPLADE query encoder with the BMP retriever.
        retr.search('my query')

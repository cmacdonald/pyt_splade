SPLADE for PyTerrier
==========================================

`pyterrier-splade <https://github.com/cmacdonald/pyt_splade>`__ is a PyTerrier extension that provides
`SPLADE <https://github.com/naver/splade>`__ learned sparse indexing and retrieval. SPLADE encodes queries
and documents into sparse bags of weighted BERT WordPiece tokens, which can be indexed and searched using standard
sparse retrieval backends such as Terrier and PISA. It was introduced in the following article:

.. cite.dblp:: conf/sigir/FormalPC21

You can install this extension with pip:

.. code-block:: bash

    pip install pyterrier-splade

.. toctree::
    :caption: Contents
    :maxdepth: 1

    Overview <overview>
    How-To Guides <how-to>
    API Reference <api>


Demo
---------------------------------------------

We have a demo of PyTerrier SPLADE at https://huggingface.co/spaces/terrierteam/splade


Credits
---------------------------------------------

 - Craig Macdonald
 - Sean MacAvaney
 - Nicola Tonellotto

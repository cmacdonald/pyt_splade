API Documentation
==========================================

:class:`~pyt_splade.Splade` is the primary way to interact with this package:

.. autoclass:: pyt_splade.Splade
    :members:

Utils
------------------------------------------

These utility transformers allow you to convert between sparse representation formats.

.. autoclass:: pyt_splade.Toks2Doc
    :members:

.. autoclass:: pyt_splade.MatchOp
    :members:

Internals
------------------------------------------

These transformers are returned by :class:`~pyt_splade.Splade` to perform encoding and scoring.

.. autoclass:: pyt_splade.SpladeEncoder
    :members:

.. autoclass:: pyt_splade.SpladeScorer
    :members:

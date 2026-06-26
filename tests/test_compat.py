import unittest


class TestBackwardCompatAlias(unittest.TestCase):
    """Ensure the legacy `pyt_splade` package name still works after the
    rename to `pyterrier_splade`."""

    def test_import_alias(self):
        import pyt_splade
        import pyterrier_splade
        self.assertIs(pyt_splade.Splade, pyterrier_splade.Splade)
        self.assertIs(pyt_splade.toks2doc, pyterrier_splade.toks2doc)
        self.assertEqual(pyt_splade.__version__, pyterrier_splade.__version__)

    def test_submodule_alias(self):
        import pyterrier_splade
        from pyt_splade._model import Splade
        self.assertIs(Splade, pyterrier_splade.Splade)


if __name__ == '__main__':
    unittest.main()

from typing import Union, List, Literal, Dict
import torch
import numpy as np
import pyterrier as pt
import pyt_splade

class Splade:
    """A SPLADE model, which provides transformers for sparse encoding documents and queries, and scoring documents."""

    def __init__(
        self,
        model: Union[torch.nn.Module, str] = "naver/splade-cocondenser-ensembledistil",
        tokenizer=None,
        agg='max',
        max_length=256,
        device=None
    ):
        """Initializes the SPLADE model.

        Args:
            model: the SPLADE model to use, either a PyTorch model or a string to load from HuggingFace
            tokenizer: the tokenizer to use, if not included in the model
            agg: the aggregation function to use for the SPLADE model
            max_length: the maximum length of the input sequences
            device: the device to use, e.g. 'cuda' or 'cpu'
        """
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

    def doc_encoder(self, text_field='text', batch_size=100, sparse=True, verbose=False, scale=100) -> pt.Transformer:
        """Returns a transformer that encodes a text field into a document representation.

        Args:
            text_field: the text field to encode
            batch_size: the batch size to use when encoding
            sparse: if True, the output will be a dict of term frequencies, otherwise a dense vector
            verbose: if True, show a progress bar
            scale: the scale to apply to the term frequencies
        """
        out_field = 'toks' if sparse else 'doc_vec'
        return pyt_splade.SpladeEncoder(self, text_field, out_field, 'd', sparse, batch_size, verbose, scale)

    indexing = doc_encoder # backward compatible name

    def query_encoder(self, batch_size=100, sparse=True, verbose=False, scale=100) -> pt.Transformer:
        """Returns a transformer that encodes a query field into a query representation.

        Args:
            batch_size: the batch size to use when encoding
            sparse: if True, the output will be a dict of term frequencies, otherwise a dense vector
            verbose: if True, show a progress bar
            scale: the scale to apply to the term frequencies
        """
        out_field = 'query_toks' if sparse else 'query_vec'
        res = pyt_splade.SpladeEncoder(self, 'query', out_field, 'q', sparse, batch_size, verbose, scale)
        return res

    query = query_encoder # backward compatible name

    def scorer(self, text_field='text', batch_size=100, verbose=False) -> pt.Transformer:
        """Returns a transformer that scores documents against queries.

        Args:
            text_field: the text field to score
            batch_size: the batch size to use when scoring
            verbose: if True, show a progress bar
        """
        return pyt_splade.SpladeScorer(self, text_field, batch_size, verbose)

    def encode(
        self,
        texts: List[str],
        rep: Literal['d', 'q'] = 'd',
        format: Literal['dict', 'np', 'torch'] ='dict',
        scale: float = 1.,
    ) -> Union[List[Dict[str, float]], List[np.ndarray], torch.Tensor]:
        """Encodes a batch of texts into their SPLADE representations.

        Args:
            texts: the list of texts to encode
            rep: 'q' for query, 'd' for document
            format: 'dict' for a dict of term frequencies, 'np' for a list of numpy arrays, 'torch' for a torch tensor
            scale: the scale to apply to the term frequencies
        """
        rtr = []
        with torch.no_grad():
            reps = self.model(**{rep + '_kwargs': self.tokenizer(
                texts,
                add_special_tokens=True,
                padding="longest",  # pad to max sequence length in batch
                truncation="longest_first",  # truncates to max model length,
                max_length=self.max_length,
                return_attention_mask=True,
                return_tensors="pt",
            ).to(self.device)})[rep + '_rep']
            reps = reps * scale
        if format == 'dict':
            reps = reps.cpu()
            for i in range(reps.shape[0]):
                # get the number of non-zero dimensions in the rep:
                col = torch.nonzero(reps[i]).squeeze(1).tolist()
                # now let's create the bow representation as a dictionary
                weights = reps[i, col].cpu().tolist()
                # if document cast to int to make the weights ready for terrier indexing
                if rep == "d":
                    weights = list(map(int, weights))
                sorted_weights = sorted(zip(col, weights), key=lambda x: (-x[1], x[0]))
                # create the dict removing the weights less than 1, i.e. 0, that are not helpful
                d = {self.reverse_voc[k]: v for k, v in sorted_weights if v > 0}
                rtr.append(d)
        elif format == 'np':
            reps = reps.cpu().numpy()
            for i in range(reps.shape[0]):
                rtr.append(reps[i])
        elif format == 'torch':
            rtr = reps
        return rtr

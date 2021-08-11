import numpy as np
import pandas as pd
from collections import Counter


class TfidfTransformer:
    def __init__(self):
        self.get_vocab_ = dict()

    @staticmethod
    def _counter(x: list)->dict:
        return Counter(x)

    def _get_vocab(self, x: list):
        for i in list(set(x)):
            self.get_vocab_[i] = x.index(i)

    def fit(self, corpus: list):
        assert isinstance(corpus, list) is True
        self._get_vocab(corpus)


class BoWTransformer:
    """Bag of words."""
    def __init__(self):
        self.model = None
        self.corpus_ = None
        self.get_vocab_ = None
        self.vocab_dict_ = dict()

    def _get_vocab_dict(self, x):
        self.vocab_dict_ = Counter(x)

    def _get_vocab_idx(self, x: list):
        for i in list(set(x)):
            self.get_vocab_[i] = x.index(i)

    def fit(self, corpus: list):
        self.corpus_ = list(set(corpus))
        self._get_vocab_dict(corpus)
        self._get_vocab_idx(self.corpus_)

    def transform(self, corpus):
        res = []
        for i in range(len(corpus)):
            res.append([(i, self.corpus_.index(corpus[i])), self.vocab_dict_[corpus[i]]])
        return res

    def fit_transform(self, corpus):
        self.fit(corpus)
        return self.transform(corpus)

    def get_feature_names(self):
        return list(self.corpus_)


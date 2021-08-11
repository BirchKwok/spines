from collections import Counter
from scipy import sparse as sp


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
        self.get_vocab_ = dict()
        self.freq_dict_ = dict()

    def _get_vocab_dict(self, x):
        self.freq_dict_ = Counter(x)

    def _get_vocab_idx(self, x: list):
        for i in list(set(x)):
            self.get_vocab_[i] = x.index(i)

    def fit(self, corpus: list):
        self.corpus_ = list(set(corpus))
        self._get_vocab_dict(corpus)
        self._get_vocab_idx(self.corpus_)

        return self

    def transform(self, corpus):
        sen_idx = [i for i in range(len(corpus))]
        word_idx = [self.corpus_.index(corpus[i]) for i in range(len(corpus))]
        word_freq = [self.freq_dict_[corpus[i]] for i in range(len(corpus))]

        X = sp.csr_matrix((word_freq, (sen_idx, word_idx)))

        return X

    def fit_transform(self, corpus):
        self.fit(corpus)
        return self.transform(corpus)





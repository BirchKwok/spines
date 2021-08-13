from collections import Counter
from scipy import sparse as sp
import re
import numpy as np
from typing import *

from spines.text._stop_words_en import STOP_WORDS_EN
from spines.text._stop_words_zh import STOP_WORDS_ZH

__all__ = [
    'BoWTransformer',
    'TfidfVectorizer'
]


def _check_stop_list(stop):
    if stop == "english":
        return STOP_WORDS_EN
    elif stop == 'chinese':
        return STOP_WORDS_ZH
    elif isinstance(stop, str):
        raise ValueError("not a built-in stop list: %s" % stop)
    elif stop is None:
        return None
    else:
        return frozenset(stop)


def _lower_en(doc):
    return doc.lower()


def _re_pattern(pattern='(?u)\b\w+\b'):
    return re.compile(pattern)


class BoWTransformer:
    """Bag of words."""

    def __init__(self, lowercase=True, token_pattern=r"(?u)\b\w+\b",
                 stop_words=None, vocabulary=None, drop_oov=True):
        self.lowercase = lowercase
        self._init_corpus = None
        self.uni_corpus_ = None
        self.vocabulary_ = list(set(vocabulary)) if vocabulary is not None else vocabulary
        self.get_corpus_idx_ = dict()
        self.get_corpus_freq_ = dict()
        self._token_pattern = _re_pattern(pattern=token_pattern)
        self._stop_words = _check_stop_list(stop_words)
        self.drop_oov = drop_oov
        self._corpus_sentence = None

    def _split_sentence(self):
        """split sentence to words"""
        return self._token_pattern.findall

    def _get_corpus_freq_dict(self, x):
        """
        count words frequence.
        return :
        a dict like {word: word frequence}
        """
        self.get_corpus_freq_ = Counter(x)

    def _get_corpus_idx(self, x: list):
        """
        get sentences index in vocabulary. 
        return :
        a dict like {sentence: sentence index}
        """
        for i in x:
            self.get_corpus_idx_[i] = x.index(i)

    def _get_doc_list(self, doc):
        """split words into list"""

        if self.lowercase:
            doc = _lower_en(doc)

        splitter = self._split_sentence()

        for i in doc:
            if '\u4e00' <= i <= '\u9fa5':  # if character is chinese
                return doc

        return splitter(doc)

    def fit(self, corpus: list):

        self._init_corpus = []
        self._corpus_sentence = []

        for i in corpus:
            tmp = self._get_doc_list(i)
            if isinstance(tmp, str):
                if self._stop_words is not None:
                    if tmp not in self._stop_words:
                        self._init_corpus.append(tmp)
                else:
                    self._init_corpus.append(tmp)
            elif isinstance(tmp, list):
                if self._stop_words is not None:
                    tmp = [i for i in tmp if i not in self._stop_words]
                self._init_corpus.extend(tmp)

            self._corpus_sentence.append(tmp)

        if self.vocabulary_ is None:
            self.uni_corpus_ = list(set(self._init_corpus))
        else:
            self.uni_corpus_ = self.vocabulary_

        self._get_corpus_freq_dict(self._init_corpus)
        self._get_corpus_idx(self.uni_corpus_)

            # self._get_corpus_freq_dict(corpus)
            # self._get_corpus_idx(self.uni_corpus_)

        return self

    def transform(self, corpus):
        assert self.uni_corpus_ is not None and self.get_corpus_idx_ is not None \
               and self.get_corpus_freq_ is not None, \
            "Not fit yet."

        c = []
        for i in corpus:
            tmp = self._get_doc_list(i)
            if isinstance(tmp, str):
                if self._stop_words is not None:
                    if tmp not in self._stop_words:
                        c.append(tmp)
                else:
                    c.append(tmp)
            else:
                if self._stop_words is not None:
                    tmp = [i for i in tmp if i not in self._stop_words]
                c.extend(tmp)

        indptr = [0]

        indices = []

        num = np.sum([len(i) if isinstance(i, Iterable) is True and isinstance(i, str) is False \
                        else 1 for i in self._corpus_sentence])
        print('num', num)
        print('c', c)

        data = [self.get_corpus_freq_[c[i]] for i in range(num)]
        print(data)
        for d in self._corpus_sentence:
            if isinstance(d, Iterable) is True and isinstance(d, str) is False:
                for term in d:
                    indices.append(self.get_corpus_idx_[term])
            else:
                indices.append(self.get_corpus_idx_[d])

            indptr.append(len(indices))

        X = sp.csr_matrix((data, indices, indptr))

        return X

    def fit_transform(self, corpus):
        self.fit(corpus)
        return self.transform(corpus)

    def inverse_transform(self, arr):
        assert isinstance(arr, list) or isinstance(arr, np.array)
        arr = np.asarray(arr)

        assert arr.ndim == 2
        arr_shape = arr.shape
        res = []
        for i in range(arr_shape[0]):
            t = []
            for j in range(len(arr[i])):
                if arr[i][j] != 0:
                    t.append(self.uni_corpus_[j])
            res.append(t)

        return res


class TfidfVectorizer(BoWTransformer):
    """TF-IDF transform from BoWTransformer"""
    def __init__(self, smooth_idf=True,
                                  ):
        super(TfidfVectorizer, self).__init__()
        self.smooth_idf = smooth_idf

    def fit(self, corpus: list):
        pass



class TfidfTransformer:

    def __init__(self):
        pass




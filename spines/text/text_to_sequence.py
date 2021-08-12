from collections import Counter
from scipy import sparse as sp
import re
import numpy as np


__all__ = [
    'BoWTransformer',
    'TfidfTransformer'
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

def _re_pattern(pattern='(?u)\b\w\w+\b'):
    return re.compile(pattern)


class BoWTransformer:
    """Bag of words."""
    def __init__(self, lowercase=True, token_pattern=r"(?u)\b\w+\b",
                stop_words=None, vocabulary=None, drop_oov=True):
        self.lowercase = lowercase
        self._init_corpus = None
        self.uni_corpus_ = None
        self.vocabulary = list(set(vocabulary)) if vocabulary is not None else vocabulary
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
            
        spliter = self._split_sentence()
        
        
        for i in doc:
            if '\u4e00' <= i <= '\u9fa5':  # if character is chinese
                return doc
            
        return spliter(doc)
         

    def fit(self, corpus: list):
        if self.vocabulary is None:
            self._init_corpus = []
            self._corpus_sentence = []
            
            for i in corpus:
                tmp = self._get_doc_list(i)
                if isinstance(tmp, str):
                    self._init_corpus.append(tmp)
                else:
                    self._init_corpus.extend(tmp)
                
                self._corpus_sentence.append(tmp)
                
            
            self.uni_corpus_ = list(set(self._init_corpus)) 
            self._get_corpus_freq_dict(self._init_corpus)    
            self._get_corpus_idx(self.uni_corpus_)
        else:
            self.uni_corpus_ = self.vocabulary 
            self._get_corpus_freq_dict(corpus)    
            self._get_corpus_idx(self.uni_corpus_)

        return self

    
    def transform(self, corpus):
        assert self.uni_corpus_ is not None and self.get_corpus_idx_ is not None and self.get_corpus_freq_ is not None, \
        "Not fit yet."
        c = []
        for i in corpus:
            tmp = self._get_doc_list(i)
            if isinstance(tmp, str):
                c.append(tmp)
            else:
                c.extend(tmp)

        
        indptr = [0]
        
        indices = []
        
        num = np.sum([len(i) for i in self._corpus_sentence])
        data = [self.get_corpus_freq_[c[i]] for i in range(num)]

        for d in self._corpus_sentence:
            for term in d:
                indices.append(self.get_corpus_idx_[term])
                
            indptr.append(len(indices))   
        
        X = sp.csr_matrix((data, indices, indptr))

        return X

    def fit_transform(self, corpus):
        self.fit(corpus)
        return self.transform(corpus)
    

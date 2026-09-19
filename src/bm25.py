"""BM25 по каталогу: разреженная матрица «документ × терм» и классическая формула.

Для русского важнее не библиотека, а морфология: слова приводятся к основе
стеммером Snowball, иначе «ремонт квартир» не совпадёт с «ремонту квартиры».
Стеммер применяется к словарю, а не к каждому вхождению: сначала матрица частот
«документ × слово», затем колонки с одинаковой основой складываются —
уникальных слов в сотни раз меньше, чем токенов (551 681 слово -> 351 023
основы на каталоге 189 212).

Токенизация — `\\w+` в нижнем регистре, так что переносы строк, эмодзи и
пунктуация отбрасываются сами собой.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import scipy.sparse as sp
import snowballstemmer
from sklearn.feature_extraction.text import CountVectorizer

TOKEN = r"(?u)\w+"


class BM25:
    name = "BM25"

    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75,
                 stemming: bool = True):
        self.stemming = stemming
        self._stemmer = snowballstemmer.stemmer("russian")

        vectorizer = CountVectorizer(lowercase=True, token_pattern=TOKEN, dtype=np.int32)
        counts = vectorizer.fit_transform(docs)
        vocab = vectorizer.get_feature_names_out()
        if stemming:
            codes, terms = pd.factorize(pd.Series(self._stemmer.stemWords(list(vocab))))
            merge = sp.csr_matrix((np.ones(len(codes), np.float32),
                                   (np.arange(len(codes)), codes)),
                                  shape=(len(vocab), len(terms)))
            tf_matrix = (counts @ merge).tocsr()
        else:
            terms, tf_matrix = pd.Index(vocab), counts.tocsr()
        self.n_words, self.n_terms = len(vocab), len(terms)
        self._term_id = pd.Series(np.arange(len(terms)), index=terms)

        doc_len = np.asarray(tf_matrix.sum(axis=1)).ravel()
        self.avg_len = doc_len.mean()
        doc_freq = np.diff(tf_matrix.tocsc().indptr)
        idf = np.log(1 + (len(docs) - doc_freq + 0.5) / (doc_freq + 0.5)).astype(np.float32)
        coo = tf_matrix.tocoo()
        tf = coo.data.astype(np.float32)
        weight = idf[coo.col] * tf * (k1 + 1) / (
            tf + k1 * (1 - b + b * doc_len[coo.row] / self.avg_len))
        self._weights_t = sp.csr_matrix((weight, (coo.row, coo.col)), shape=tf_matrix.shape,
                                        dtype=np.float32).T.tocsr()
        self._texts: np.ndarray | None = None

    def query_matrix(self, texts) -> sp.csr_matrix:
        """Разреженная матрица «запрос × терм» той же формы, что и индекс."""
        rows, cols = [], []
        for i, text in enumerate(texts):
            tokens = re.findall(TOKEN, str(text).lower())
            if self.stemming:
                tokens = self._stemmer.stemWords(tokens)
            for token in tokens:
                j = self._term_id.get(token, -1)
                if j >= 0:
                    rows.append(i)
                    cols.append(j)
        return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                             shape=(len(texts), self.n_terms))

    # интерфейс ретривера (см. retrieve.py)
    def prepare(self, texts: np.ndarray) -> None:
        self._texts = texts

    def scores(self, rows: slice) -> np.ndarray:
        """Скоры запросов `texts[rows]` по всему каталогу: (n_запросов, n_документов)."""
        return (self.query_matrix(self._texts[rows]) @ self._weights_t).toarray()

    def pair_scores(self, query_rows: np.ndarray, items: np.ndarray,
                    batch: int = 200_000) -> np.ndarray:
        """Скоры отдельных пар: запрос `texts[query_rows[i]]` — документ `items[i]`."""
        q = self.query_matrix(self._texts)
        docs = self._weights_t.T.tocsr()
        out = np.empty(len(items), dtype=np.float32)
        for s in range(0, len(items), batch):
            e = s + batch
            out[s:e] = np.asarray(q[query_rows[s:e]].multiply(docs[items[s:e]]).sum(axis=1)).ravel()
        return out

"""Поиск топ-K по каталогу: отсечение кандидатов, ранжирование ретриверами, RRF.

Один и тот же алгоритм используется на валидации (`evaluate.py`) и на бенчмарке
(`run/retrieve_benchmark.py`).

Ретривер — объект с атрибутом `name` и двумя методами:

* `prepare(texts)` — получить уникальные тексты запросов (закодировать и т.п.);
* `scores(rows)` — скоры запросов `texts[rows]` по всему каталогу, массив
  (n_запросов, n_документов).

Скоры считаются один раз на уникальный текст запроса и переиспользуются всеми
запросами с этим текстом. Для каждого пула кандидатов (`pools.py`) ретривер
отдаёт первые `rrf_depth` кандидатов по убыванию скора: их топ-K — ответ
ретривера, а сами списки сливаются в RRF (Reciprocal Rank Fusion):

    RRF(d) = Σ_r 1 / (rrf_k + rank_r(d))

с равными весами ретриверов. RRF смотрит только на ранги, так что шкалы BM25 и
косинуса согласовывать не нужно. Ранги считаются внутри пула, после отсечения.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from logs import step
from pools import NEIGHBORS, VARIANTS, CandidatePools, filter_key

FUSED = "RRF"


def build_queries(df: pd.DataFrame, pools: CandidatePools, id_col: str,
                  extra: tuple[str, ...] = ()) -> pd.DataFrame:
    """Запросы для поиска: текст, комбинация фильтров и гео, индекс — `id_col`.

    Строки с одинаковым `id_col` схлопываются (первая), `extra` — дополнительные колонки.
    """
    df = df.assign(fkey=filter_key(df), geo=pools.geo_codes(df.search_location_id),
                   nb=pools.neighbor_codes(df.search_location_id))
    agg = {"query": ("search_query", "first"), "fkey": ("fkey", "first"),
           "geo": ("geo", "first"), "nb": ("nb", "first"), **{c: (c, "first") for c in extra}}
    return df.groupby(id_col, sort=False).agg(**agg)


def ranking(score: np.ndarray, pool: np.ndarray | None, depth: int) -> np.ndarray:
    """Первые depth кандидатов пула по убыванию скора — индексы каталога."""
    sub = score if pool is None else score[pool]
    d = min(depth, len(sub))
    if d == 0:
        return np.empty(0, dtype=np.int64)
    top = np.argpartition(-sub, d - 1)[:d] if d < len(sub) else np.arange(len(sub))
    top = top[np.argsort(-sub[top], kind="stable")]
    return top if pool is None else pool[top]


def rrf(rankings: list[np.ndarray], k: int, top_k: int) -> np.ndarray:
    idx = np.concatenate(rankings)
    weight = np.concatenate([1.0 / (k + np.arange(1, len(r) + 1)) for r in rankings])
    uniq, inv = np.unique(idx, return_inverse=True)
    fused = np.bincount(inv, weights=weight)
    return uniq[np.argsort(-fused, kind="stable")[:top_k]]


def fill_top(main, extra, top_k: int) -> list:
    """Ответ основного варианта, добитый до top_k кандидатами запасного без повторов.

    Порядок на Recall@K не влияет, поэтому добивка не снижает recall ни одного запроса.
    """
    main = list(main)
    seen = set(main)
    return main + [x for x in extra if x not in seen][:max(0, top_k - len(main))]


def result_names(retrievers: list, fuse: bool) -> list[str]:
    names = [r.name for r in retrievers]
    return names + ([FUSED] if fuse and len(retrievers) > 1 else [])


def search(queries: pd.DataFrame, pools: CandidatePools, retrievers: list, top_k: int,
           variants=VARIANTS, fuse: bool = True, rrf_k: int = 60, rrf_depth: int = 200,
           chunk: int = 256) -> Iterator[tuple[object, str, dict[str, np.ndarray]]]:
    """Для каждого запроса и варианта отсечения — топ-K каждого ретривера и RRF.

    Выдаёт `(id запроса, вариант, {ретривер: индексы каталога по убыванию})`.
    В топе может быть меньше K объявлений, если пул меньше K.
    """
    names = [r.name for r in retrievers]
    columns = result_names(retrievers, fuse)
    fuse = FUSED in columns
    texts = queries["query"].unique()
    for r in retrievers:
        with step(f"{r.name}: готовим {len(texts):,} уникальных запросов"):
            r.prepare(texts)

    ids_of_text = queries.groupby("query", sort=False).groups
    message = f"ищем топ-{top_k}: {', '.join(columns)} × отсечение: {', '.join(variants)}"
    with step(message), tqdm(total=len(texts), unit="текст", desc="поиск") as bar:
        for start in range(0, len(texts), chunk):
            rows = slice(start, min(start + chunk, len(texts)))
            scores = [r.scores(rows) for r in retrievers]
            for i, text in enumerate(texts[rows]):
                group = queries.loc[ids_of_text[text]]
                for v in variants:
                    done = {}     # запросы с одним текстом и одним пулом делят результат
                    geos = group.nb if v == NEIGHBORS else group.geo
                    for qid, fkey, geo in zip(group.index, group.fkey, geos):
                        key = pools.key(v, fkey, geo)
                        if key not in done:
                            pool = pools.indices(key)
                            ranks = [ranking(s[i], pool, rrf_depth) for s in scores]
                            tops = {n: rk[:top_k] for n, rk in zip(names, ranks)}
                            if fuse:
                                tops[FUSED] = rrf(ranks, rrf_k, top_k)
                            done[key] = tops
                        yield qid, v, done[key]
            bar.update(rows.stop - rows.start)

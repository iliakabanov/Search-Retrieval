"""Кандидаты для переранжировщика: K объявлений на событие от текущего пайплайна.

Кандидаты — RRF (BM25 + dense) по пулам (`retrieve.search`): топ внутри гео
запроса, топ по соседним локациям города (до 50 км, отдельная квота) и лучшие из
«только фильтры», которых ещё нет (`retrieve.fill_top`). Квота под соседей
поднимает потолок кандидатов на валидации с 95.1% до 96.5%: треть позитивов,
не попадавших в кандидаты, лежала в соседней локации. На валидации при K = 300 и
квоте 20% среди кандидатов 95.1% позитивов против 87.9% в нынешнем топ-50 —
это потолок для переранжировщика (docs/reranker_plan.md).

Для каждого кандидата сохраняется «сырьё» для признаков: позиция в итоговом
списке, откуда он (гео или квота) и ранг в каждом ретривере и в RRF внутри
обоих пулов (-1 — не попал в их топ-K). Скоры ретриверов досчитываются на шаге
признаков.

Для обучения на событие оставляются все позитивы, `neg_top` самых высоких
негативов и `neg_random` случайных из остальных: все K × 173 тыс. событий не
помещаются в память, а трудные негативы важнее случайных.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd

from pools import NEIGHBORS, CandidatePools
from retrieve import FUSED, fill_top, search

GEO, EXTRA = "только гео", "только фильтры"
SOURCE_GEO, SOURCE_QUOTA, SOURCE_NEIGHBORS = 0, 1, 2
#: Пулы кандидатов и суффиксы их колонок рангов.
PARTS = {GEO: "geo", NEIGHBORS: "nb", EXTRA: "extra"}


def generate(queries: pd.DataFrame, pools: CandidatePools, retrievers: list, k: int = 300,
             quota: float = 0.2, nb_quota: float = 0.0, gold: dict | None = None,
             neg_top: int | None = None, neg_random: int | None = None, seed: int = 0,
             chunk_queries: int = 5_000, rrf_depth: int | None = None) -> Iterator[pd.DataFrame]:
    """Кандидаты по запросам, порциями DataFrame.

    Места: `1 - quota - nb_quota` — топ RRF внутри гео, `nb_quota` — топ RRF по
    соседним локациям города (`pools.neighbors`), остальные — вне гео («только
    фильтры»). Незанятые места (мало соседей, маленький пул) достаются следующему пулу.

    Колонки: `qid`, `item_idx` (индекс в корпусе), `cand_rank` (позиция в списке),
    `source` (0 — гео, 2 — соседи, 1 — вне гео), ранги ретриверов и RRF в каждом
    пуле `rank_<ретривер>_<geo|nb|extra>` (-1 — не в топ-k пула), `label` (если
    передан `gold`: {qid: множество индексов позитивов}).
    `neg_top` / `neg_random` — прореживание негативов для обучения (нужен gold).
    `rrf_depth` — сколько лучших от каждого ретривера сливает RRF (по умолчанию k):
    на валидации глубина 500 при k = 300 поднимает recall кандидатов на 1.1 п.п.
    """
    names = [r.name for r in retrievers] + [FUSED]
    variants = [GEO, NEIGHBORS, EXTRA] if nb_quota > 0 else [GEO, EXTRA]
    k_geo = int(round(k * (1 - quota - nb_quota)))
    k_nb = int(round(k * nb_quota))
    rng = np.random.default_rng(seed)
    pending: dict = {}
    rows = []

    for qid, variant, tops in search(queries, pools, retrievers, k, variants, True,
                                     rrf_depth=max(rrf_depth or k, k)):
        got = pending.setdefault(qid, {})
        got[variant] = tops
        if len(got) < len(variants):
            continue
        del pending[qid]

        geo_part = got[GEO][FUSED][:k_geo].tolist()
        with_nb = fill_top(geo_part, got[NEIGHBORS][FUSED].tolist(), k_geo + k_nb)             if nb_quota > 0 else geo_part
        cand = np.asarray(fill_top(with_nb, got[EXTRA][FUSED].tolist(), k), dtype=np.int64)
        index = pd.Index(cand)
        pos_ = np.arange(len(cand))
        row = {"qid": np.full(len(cand), qid, dtype=object), "item_idx": cand,
               "cand_rank": pos_.astype(np.int16),
               "source": np.select([pos_ < len(geo_part), pos_ < len(with_nb)],
                                   [SOURCE_GEO, SOURCE_NEIGHBORS], SOURCE_QUOTA).astype(np.int8)}
        for n in names:
            for variant_, part in PARTS.items():
                if variant_ not in got:
                    continue
                ranks = np.full(len(cand), -1, dtype=np.int16)
                pos = index.get_indexer(got[variant_][n])
                found = pos >= 0
                ranks[pos[found]] = np.flatnonzero(found)
                row[f"rank_{n}_{part}"] = ranks

        if gold is not None:
            label = np.isin(cand, list(gold[qid])).astype(np.int8)
            row["label"] = label
            if neg_top is not None:
                neg = np.flatnonzero(label == 0)
                rest = neg[neg_top:]
                keep = np.concatenate([np.flatnonzero(label == 1), neg[:neg_top],
                                       rng.choice(rest, min(neg_random or 0, len(rest)),
                                                  replace=False)])
                row = {c: v[np.sort(keep)] for c, v in row.items()}
        rows.append(pd.DataFrame(row))

        if len(rows) >= chunk_queries:
            yield pd.concat(rows, ignore_index=True)
            rows = []
    if rows:
        yield pd.concat(rows, ignore_index=True)

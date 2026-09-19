"""Кандидаты для переранжировщика: K объявлений на событие от текущего пайплайна.

Кандидаты — RRF (BM25 + dense) по двум пулам (`retrieve.search`): доля
`1 - quota` мест — топ внутри гео запроса, остальные — лучшие из «только
фильтры», которых ещё нет (`retrieve.fill_top`). На валидации при K = 300 и
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

from pools import CandidatePools
from retrieve import FUSED, fill_top, search

GEO, EXTRA = "только гео", "только фильтры"
SOURCE_GEO, SOURCE_QUOTA = 0, 1


def rank_columns(names: list[str]) -> list[str]:
    """Колонки рангов: ret_<ретривер>_<geo|extra>."""
    return [f"rank_{n}_{p}" for n in names for p in ("geo", "extra")]


def generate(queries: pd.DataFrame, pools: CandidatePools, retrievers: list, k: int = 300,
             quota: float = 0.2, gold: dict | None = None, neg_top: int | None = None,
             neg_random: int | None = None, seed: int = 0, chunk_queries: int = 5_000
             ) -> Iterator[pd.DataFrame]:
    """Кандидаты по запросам, порциями DataFrame.

    Колонки: `qid`, `item_idx` (индекс в корпусе), `cand_rank` (позиция в списке),
    `source` (0 — гео, 1 — квота), ранги ретриверов и RRF по пулам, `label`
    (если передан `gold`: {qid: множество индексов позитивов}).
    `neg_top` / `neg_random` — прореживание негативов для обучения (нужен gold).
    """
    names = [r.name for r in retrievers] + [FUSED]
    k_geo = int(round(k * (1 - quota)))
    rng = np.random.default_rng(seed)
    pending, rows = {}, []

    for qid, variant, tops in search(queries, pools, retrievers, k, [GEO, EXTRA], True,
                                     rrf_depth=k):
        other = pending.pop((qid, EXTRA if variant == GEO else GEO), None)
        if other is None:
            pending[(qid, variant)] = tops
            continue
        geo, extra = (tops, other) if variant == GEO else (other, tops)

        geo_part = geo[FUSED][:k_geo].tolist()
        cand = np.asarray(fill_top(geo_part, extra[FUSED].tolist(), k), dtype=np.int64)
        index = pd.Index(cand)
        row = {"qid": np.full(len(cand), qid, dtype=object), "item_idx": cand,
               "cand_rank": np.arange(len(cand), dtype=np.int16),
               "source": np.where(np.arange(len(cand)) < len(geo_part),
                                  SOURCE_GEO, SOURCE_QUOTA).astype(np.int8)}
        for n in names:
            for part, result in (("geo", geo), ("extra", extra)):
                ranks = np.full(len(cand), -1, dtype=np.int16)
                pos = index.get_indexer(result[n])
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

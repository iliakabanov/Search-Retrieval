"""Оценка ретриверов на валидации: recall@K при каждом варианте отсечения.

Поиск — `retrieve.search` (тот же алгоритм, что на бенчмарке): для каждого
события и варианта отсечения берётся топ-K каждого ретривера и их RRF, и
считается, сколько позитивов события в него попало.

Первые результаты (recall@50, каталог 189 212, 32 891 событие валидации,
dense-модели без дообучения, `python run/evaluate_retrievers.py`):

| | без отсечения | только фильтры | только гео | фильтры + гео | гео + добивка |
|---|---|---|---|---|---|
| потолок | 100% | 96.9% | 81.0% | 78.7% | 99.3% |
| случайный порядок | 0.03% | 1.4% | 13.8% | 43.5% | — |
| BM25 | 24.7% | 26.1% | 74.9% | 75.6% | 79.8% |
| e5-large | 25.6% | 26.8% | 76.4% | 76.0% | 81.3% |
| RoSBERTa | 22.3% | 23.8% | 74.6% | 75.0% | 78.9% |
| RRF (все три) | 27.8% | 28.9% | 78.1% | 77.0% | 83.5% |

* Добивка топа «только гео» до 50 лучшими из «только фильтры» даёт +4.3…+5.4 п.п.
  всем ретриверам: у ~20% событий пул города меньше 50, и свободные места
  занимают кандидаты из других городов, где лежит часть из 19% отсечённых позитивов.

* Качество определяет город: тот же BM25 без гео даёт 25%, в пуле города — 75%.
  Похоже, без гео топ заполняют подходящие по смыслу объявления из других
  городов (гипотеза, проверяется по доле города запроса в топе).
* Внутри города ранжирование близко к потолку (RRF — 96% от 81.0%); главный
  резерв — 19% позитивов, которые жёсткое гео отсекает.
* e5 без дообучения лишь на 0.7–1.4 п.п. лучше BM25, RoSBERTa — хуже BM25. Но
  ошибаются они на разном: ~10% событий находит только BM25 и ~10% — только e5,
  отсюда +2–3 п.п. у RRF. RoSBERTa добавляет к RRF(BM25 + e5) не больше 0.4 п.п.
* Срез «знакомый текст» без гео хуже «нового» на ~5 п.п. у всех ретриверов, в
  том числе у BM25 без обучения: это состав запросов (частые запросы — много
  однотипных объявлений в других городах), а не запоминание.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pools import FILL_EXTRA, FILL_MAIN, FILLED, REPORT_VARIANTS, VARIANTS, CandidatePools
from retrieve import build_queries, fill_top, result_names, search


def build_events(val: pd.DataFrame, pools: CandidatePools, item_ids: pd.Series) -> pd.DataFrame:
    """События валидации: текст, фильтры, город и множество позитивов (индексы каталога)."""
    events = build_queries(val, pools, "event_id", extra=("slice",)).sort_index()
    idx = val.item_id.map(pd.Series(np.arange(len(item_ids)), index=item_ids))
    events["positives"] = idx.groupby(val.event_id).apply(lambda s: set(s.to_numpy()))
    events["n_pos"] = events.positives.map(len)
    return events


def evaluate(events: pd.DataFrame, pools: CandidatePools, retrievers: list, top_k: int,
             fuse: bool = True, rrf_k: int = 60, rrf_depth: int = 200,
             chunk: int = 256) -> pd.DataFrame:
    """Число найденных позитивов в топ-K по событиям.

    Колонки — MultiIndex (ретривер, вариант); RRF добавляется, если ретриверов > 1.
    Кроме четырёх вариантов отсечения считается комбинация `FILLED`: топ «только
    гео», добитый до K из «только фильтры», — так собирается ответ на бенчмарке.
    """
    columns = result_names(retrievers, fuse)
    row_of_event = pd.Series(np.arange(len(events)), index=events.index)
    gold = events.positives.to_dict()
    hits = {(n, v): np.zeros(len(events)) for n in columns for v in REPORT_VARIANTS}
    pending = {}    # топы одного из двух вариантов для добивки, пока не пришёл второй
    for event_id, v, tops in search(events, pools, retrievers, top_k, VARIANTS, fuse,
                                    rrf_k, rrf_depth, chunk):
        pos = row_of_event[event_id]
        for n, top in tops.items():
            hits[(n, v)][pos] = len(gold[event_id].intersection(top.tolist()))
        if v in (FILL_MAIN, FILL_EXTRA):
            other = pending.pop((event_id, FILL_EXTRA if v == FILL_MAIN else FILL_MAIN), None)
            if other is None:
                pending[(event_id, v)] = tops
                continue
            main, extra = (tops, other) if v == FILL_MAIN else (other, tops)
            for n in main:
                filled = fill_top(main[n].tolist(), extra[n].tolist(), top_k)
                hits[(n, FILLED)][pos] = len(gold[event_id].intersection(filled))

    out = pd.DataFrame(hits, index=events.index)
    out.columns = pd.MultiIndex.from_tuples(out.columns, names=["ретривер", "вариант"])
    return out


# --- сводные таблицы ---------------------------------------------------------


def per_event_recall(hits: pd.DataFrame, events: pd.DataFrame, cutoffs: pd.DataFrame
                     ) -> pd.DataFrame:
    """Recall по событиям: ретриверы, случайный порядок и потолок."""
    n_pos = events.n_pos.to_numpy()[:, None]
    recall = hits / n_pos
    for q, name in (("random_hits", "случайный порядок"), ("passed", "потолок")):
        extra = cutoffs.xs(q, axis=1, level=1)[list(REPORT_VARIANTS)] / n_pos
        extra.columns = pd.MultiIndex.from_product([[name], extra.columns])
        recall = pd.concat([extra, recall], axis=1)
    recall.columns.names = ["ретривер", "вариант"]
    return recall


def recall_table(recall: pd.DataFrame) -> pd.DataFrame:
    """Строки — ретриверы, столбцы — варианты отсечения."""
    # порядок строк — как в recall: потолок, случайный порядок, ретриверы, RRF
    names = list(recall.columns.get_level_values(0).unique())
    return recall.mean().unstack("вариант").loc[names, list(REPORT_VARIANTS)]


def hit_rate_table(recall: pd.DataFrame) -> pd.DataFrame:
    """Доля событий, где в топе есть хотя бы один позитив."""
    names = [n for n in recall.columns.get_level_values(0).unique()
             if n not in ("потолок", "случайный порядок")]
    return (recall[names] > 0).mean().unstack("вариант")[list(REPORT_VARIANTS)].loc[names]


def slice_table(recall: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Recall по срезам валидации «новый / знакомый текст»."""
    names = [n for n in recall.columns.get_level_values(0).unique()
             if n not in ("потолок", "случайный порядок")]
    table = recall[names].groupby(events["slice"]).mean().T
    table["разница"] = table["знакомый текст"] - table["новый текст"]
    return table.swaplevel().sort_index(level=0, sort_remaining=False).reindex(
        list(REPORT_VARIANTS), level=0)


def complementarity(recall: pd.DataFrame, names: list[str], variant: str) -> pd.DataFrame:
    """Доля событий: нашёл ретривер в строке и не нашёл ретривер в столбце."""
    found = {n: recall[(n, variant)] > 0 for n in names}
    return pd.DataFrame({b: {a: (found[a] & ~found[b]).mean() for a in names} for b in names})

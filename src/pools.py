"""Отсечение кандидатов до ранжирования: фильтры запроса и гео.

Четыре варианта (`VARIANTS`): без отсечения, только фильтры запроса, только гео
(`search_location_id == item_location_id`) и оба вместе.

Маски фильтров повторяют `infm_params.match_filters`: пустое поле объявления —
несоответствие, метки «No category» / «No subcategory» совпадают только с такими
же метками в фильтре.

**Потолок recall** — доля позитивов, переживших отсечение: предел для любого
ранжирования внутри пула. **Случайный порядок** — ожидаемый recall@K без
ранжирования: из `p` переживших позитивов в топ-K пула размера `c` попадает в
среднем `K·p/c` (или все `p`, если `c ≤ K`).

Что показали эти числа на валидации (каталог 189 212):

| вариант | потолок | пул, медиана | случайный порядок |
|---|---|---|---|
| без отсечения | 100% | 189 212 | 0.03% |
| только фильтры | 96.9% | 21 015 | 1.4% |
| только гео | 81.0% | 751 | 13.8% |
| фильтры + гео | 78.7% | 45 | 43.5% |

Фильтры почти ничего не теряют, но и сжимают слабо: у 34.8% событий фильтров
нет. Гео сжимает пул в сотни раз, но теряет 19% позитивов — у 17% пар город
запроса и объявления не совпадают. Поэтому гео как жёсткое отсечение ограничивает
recall сверху; его разумнее отдавать ранжированию как признак.

Для потолка и случайного порядка сами пулы не нужны — только их размеры
(`stats`). Индексы кандидатов (`indices`) строятся лениво при оценке и
кэшируются: пул зависит от комбинации фильтров и города, а таких комбинаций куда
меньше, чем событий.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

VARIANTS = ("без отсечения", "только фильтры", "только гео", "фильтры + гео")

#: Не отдельный пул, а комбинация: топ «только гео», добитый до K лучшими из
#: «только фильтры» (см. retrieve.fill_top). Так собирается ответ на бенчмарке.
FILLED, FILL_MAIN, FILL_EXTRA = "гео + добивка", "только гео", "только фильтры"
REPORT_VARIANTS = VARIANTS + (FILLED,)


def filter_key(df: pd.DataFrame) -> list[tuple]:
    """Комбинация фильтров события: (вид, типы, предметы, онлайн-запись)."""
    return list(zip(df.filter_category.fillna(""),
                    df.filter_subcategory.map(lambda x: tuple(sorted(x))),
                    df.filter_subject.map(lambda x: tuple(sorted(x))),
                    df.filter_online_booking))


class CandidatePools:
    """Пулы кандидатов по каталогу `items` (атрибуты из `data.FILTER_ITEM_COLUMNS`)."""

    def __init__(self, items: pd.DataFrame):
        self.n_items = len(items)
        self._category = items.item_category.fillna("").to_numpy()
        self._subcategory = items.item_subcategory.fillna("").to_numpy()
        self._subjects = items.item_subjects.map(tuple).to_numpy()
        self._booking = items.item_online_booking.to_numpy()
        self.loc_code, self._loc_values = pd.factorize(items.item_location_id)
        self._loc_index = pd.Series(np.arange(len(self._loc_values)), index=self._loc_values)
        self._geo_size = np.bincount(self.loc_code, minlength=len(self._loc_values))
        self.masks: dict[tuple, np.ndarray] = {}
        self._both_size: dict[tuple, np.ndarray] = {}
        self._cache: dict[tuple, np.ndarray | None] = {}

    def location_codes(self, location_ids: pd.Series) -> np.ndarray:
        """Код города каталога; -1, если в каталоге нет объявлений из этого города."""
        return location_ids.map(self._loc_index).fillna(-1).astype(int).to_numpy()

    def mask(self, fkey: tuple) -> np.ndarray:
        if fkey not in self.masks:
            category, subcategory, subject, booking = fkey
            m = np.ones(self.n_items, dtype=bool)
            if category:
                m &= self._category == category
            if subcategory:
                m &= np.isin(self._subcategory, subcategory)
            if subject:
                m &= np.array([bool(set(s) & set(subject)) for s in self._subjects])
            if booking:
                m &= self._booking
            self.masks[fkey] = m
        return self.masks[fkey]

    @staticmethod
    def key(variant: str, fkey: tuple, loc: int) -> tuple:
        """События с одинаковым ключом делят один пул."""
        return {"без отсечения": (variant,), "только фильтры": (variant, fkey),
                "только гео": (variant, loc), "фильтры + гео": (variant, fkey, loc)}[variant]

    def indices(self, key: tuple) -> np.ndarray | None:
        """Индексы кандидатов в каталоге; None — весь каталог."""
        if key not in self._cache:
            variant, *rest = key
            if variant == "без отсечения":
                pool = None
            elif variant == "только фильтры":
                pool = np.flatnonzero(self.mask(rest[0]))
            elif variant == "только гео":
                pool = np.flatnonzero(self.loc_code == rest[0])
            else:
                pool = np.flatnonzero(self.mask(rest[0]) & (self.loc_code == rest[1]))
            self._cache[key] = pool
        return self._cache[key]

    def stats(self, fkey: tuple, loc: int, gold: np.ndarray) -> dict[str, tuple[int, int]]:
        """{вариант: (позитивов пережило отсечение, размер пула)} для одного события."""
        m = self.mask(fkey)
        if fkey not in self._both_size:
            self._both_size[fkey] = np.bincount(self.loc_code[m],
                                                minlength=len(self._loc_values))
        in_filters = m[gold]
        in_geo = self.loc_code[gold] == loc     # loc = -1 не совпадёт ни с чем
        return {
            "без отсечения": (len(gold), self.n_items),
            "только фильтры": (int(in_filters.sum()), int(m.sum())),
            "только гео": (int(in_geo.sum()), int(self._geo_size[loc]) if loc >= 0 else 0),
            "фильтры + гео": (int((in_filters & in_geo).sum()),
                              int(self._both_size[fkey][loc]) if loc >= 0 else 0),
            # кандидаты добивки — из пулов гео и фильтров; размер пула не определён
            FILLED: (int((in_filters | in_geo).sum()), -1),
        }


def cutoff_stats(events: pd.DataFrame, pools: CandidatePools, top_k: int) -> pd.DataFrame:
    """По событиям и вариантам: пережившие позитивы, размер пула, случайный порядок.

    Колонки — MultiIndex (вариант, величина), величины: `passed`, `pool`, `random_hits`.
    """
    rows = []
    for fkey, loc, gold in zip(events.fkey, events.loc_code, events.positives):
        gold = np.fromiter(gold, dtype=np.int64)
        rows.append({(v, q): x for v, (n_pass, n_cand) in pools.stats(fkey, loc, gold).items()
                     for q, x in (("passed", n_pass), ("pool", n_cand))})
    out = pd.DataFrame(rows, index=events.index)
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    for v in VARIANTS:
        passed, pool = out[(v, "passed")], out[(v, "pool")]
        out[(v, "random_hits")] = np.where(pool <= top_k, passed,
                                           top_k * passed / np.maximum(pool, 1))
    out[(FILLED, "random_hits")] = np.nan      # у комбинации нет единого пула
    return out

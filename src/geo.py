"""Гео для запросов с регионом вместо города.

У части запросов `search_location_id` — не город, а, судя по данным, регион:
такая локация ни разу не встречается как `item_location_id` (ни в корпусе, ни в
train), поэтому пул «только гео» у них пустой, и весь ответ — добивка поиском
без гео. Таких запросов 11.5% пар train, 12% событий валидации и 17.3% запросов
бенчмарка; recall у них ~40% против ~86–89% у запросов с городом.

Позитивы таких запросов лежат в конкретных локациях. Регионы бывают двух типов:
с явным центром (у 107621 в одной локации 79% позитивов, у 107620 — 61%) и
размазанные по стране (у 621540 топ-1 даёт 27%, для 80% нужно 65 локаций).

`region_map` строит по парам train соответствие «регион → локации его
позитивов»: самые частые локации, пока их доля не наберёт `cover`. Для
валидации соответствие строится только по `train_pairs`, без утечки.
"""

from __future__ import annotations

import pandas as pd


def region_map(pairs: pd.DataFrame, item_locations: set, cover: float) -> dict[int, list[int]]:
    """{регион: локации объявлений}, покрывающие долю `cover` его позитивов.

    `pairs` — колонки `search_location_id`, `item_location_id`;
    регион — локация запроса, которой нет среди `item_locations`.
    """
    if cover <= 0:
        return {}
    regional = pairs[~pairs.search_location_id.isin(item_locations)]
    result = {}
    for region, locs in regional.groupby("search_location_id").item_location_id:
        share = locs.value_counts(normalize=True)
        n = int((share.cumsum() < cover - 1e-9).sum()) + 1
        result[region] = share.index[:n].tolist()
    return result

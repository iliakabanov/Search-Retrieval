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


def corpus_centroids(items: pd.DataFrame) -> pd.DataFrame:
    """Центры локаций (медиана координат объявлений) по всему train и корпусу `items`."""
    import pyarrow.parquet as pq

    from features import location_centroids
    from paths import PROCESSED

    tr = pq.read_table(PROCESSED / "train.parquet", columns=[
        "item_location_id", "item_latitude", "item_longitude"]).to_pandas()
    return location_centroids(pd.concat([tr.item_location_id, items.item_location_id]),
                              pd.concat([tr.item_latitude, items.item_latitude]),
                              pd.concat([tr.item_longitude, items.item_longitude]))


def neighbor_map(centroids: pd.DataFrame, radius_km: float) -> dict[int, list[int]]:
    """{локация: соседние локации}, чьи центры не дальше `radius_km` от её центра.

    `centroids` — индекс локация, колонки lat, lon (`features.location_centroids`).
    Соседние локации — это пригороды и города агломерации: на валидации 35% позитивов,
    не попавших в кандидатов, лежат в другой локации не дальше 50 км от города запроса.
    """
    if radius_km <= 0:
        return {}
    import numpy as np

    locs = centroids.index.to_numpy()
    lat, lon = np.radians(centroids.lat.to_numpy()), np.radians(centroids.lon.to_numpy())
    result = {}
    for i, loc in enumerate(locs):
        h = (np.sin((lat - lat[i]) / 2) ** 2
             + np.cos(lat[i]) * np.cos(lat) * np.sin((lon - lon[i]) / 2) ** 2)
        dist = 2 * 6371 * np.arcsin(np.sqrt(h))
        near = np.flatnonzero((dist <= radius_km) & (locs != loc))
        if len(near):
            result[int(loc)] = [int(x) for x in locs[near[np.argsort(dist[near])]]]
    return result


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

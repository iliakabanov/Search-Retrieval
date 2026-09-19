"""Загрузка разбиения из `dataset/split/` (см. `split.py`)."""

from __future__ import annotations

import json

import pandas as pd
import pyarrow.parquet as pq

from paths import PROCESSED, SPLIT

#: Атрибуты объявления, по которым работают фильтры запроса и гео.
FILTER_ITEM_COLUMNS = ["item_category", "item_subcategory", "item_subjects",
                       "item_online_booking", "item_location_id"]
TEXT_ITEM_COLUMNS = ["item_title_raw", "item_description_raw"]


def load_split_config() -> dict:
    return json.loads((SPLIT / "config.json").read_text(encoding="utf-8"))


def load_catalog(columns: list[str]) -> pd.DataFrame:
    """Объявления каталога в его порядке, с атрибутами из обработанного train.

    Порядок строк — порядок `catalog.parquet`: в нём же лежат эмбеддинги.
    """
    catalog_ids = pq.read_table(SPLIT / "catalog.parquet").to_pandas().item_id
    columns = ["item_id"] + [c for c in columns if c != "item_id"]
    return (pq.read_table(PROCESSED / "train.parquet", columns=columns).to_pandas()
            .drop_duplicates("item_id").set_index("item_id").loc[catalog_ids].reset_index())


def load_item_locations() -> pd.Series:
    """Локация каждого объявления train: item_id -> item_location_id."""
    df = pq.read_table(PROCESSED / "train.parquet",
                       columns=["item_id", "item_location_id"]).to_pandas()
    return df.drop_duplicates("item_id").set_index("item_id").item_location_id


def load_geo_pairs(train_pairs_only: bool) -> pd.DataFrame:
    """Пары (search_location_id, item_location_id) для geo.region_map.

    Для валидации — только обучающая часть разбиения (`train_pairs`), чтобы не было
    утечки; для бенчмарка — весь обработанный train.
    """
    if train_pairs_only:
        pairs = pq.read_table(SPLIT / "train_pairs.parquet",
                              columns=["search_location_id", "item_id"]).to_pandas()
        pairs["item_location_id"] = pairs.item_id.map(load_item_locations())
        return pairs[["search_location_id", "item_location_id"]]
    return pq.read_table(PROCESSED / "train.parquet",
                         columns=["search_location_id", "item_location_id"]).to_pandas()


def load_val_eval() -> pd.DataFrame:
    """Позитивы событий валидации, готовых к оценке."""
    return pq.read_table(SPLIT / "val_eval.parquet").to_pandas()


def load_benchmark_items(columns: list[str]) -> pd.DataFrame:
    """Корпус бенчмарка — объявления, среди которых ищем ответы на запросы бенчмарка."""
    columns = ["item_id"] + [c for c in columns if c != "item_id"]
    items = pq.read_table(PROCESSED / "benchmark_items.parquet", columns=columns).to_pandas()
    return items.drop_duplicates("item_id").reset_index(drop=True)


def load_train_extra_items(columns: list[str]) -> pd.DataFrame:
    """Объявления train, которых нет в каталоге валидации (~155 тыс.), по item_id.

    Вместе с каталогом это все объявления train — корпус для обучения
    переранжировщика на всех событиях train_pairs (docs/reranker_plan.md, вариант 2).
    """
    catalog = set(pq.read_table(SPLIT / "catalog.parquet").to_pandas().item_id)
    columns = ["item_id"] + [c for c in columns if c != "item_id"]
    items = (pq.read_table(PROCESSED / "train.parquet", columns=columns).to_pandas()
             .drop_duplicates("item_id"))
    return items[~items.item_id.isin(catalog)].sort_values("item_id").reset_index(drop=True)


def load_benchmark_queries() -> pd.DataFrame:
    return pq.read_table(PROCESSED / "benchmark_queries.parquet").to_pandas()

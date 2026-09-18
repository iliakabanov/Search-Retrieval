"""Разбиение обработанного train на обучение и валидацию, каталог для поиска.

Шаги (все параметры — в `SplitConfig`):

1. **Дедупликация.** Ключ примера — `(событие, item_id)`, где событие — это
   поля запроса из `benchmark_queries`: текст, гео и фильтры. Это то же, что
   дедуп по всей строке: поля объявления однозначно заданы `item_id`. Число
   повторов сохраняется в `n_repeats`, чтобы вес примера остался явным.
   497 673 строки -> 464 933 примера, 345 135 событий, 74 529 текстов.

2. **Разбиение по тексту запроса**, а не по событию: иначе «маникюр» из Казани
   попадёт в обучение, а «маникюр» из Уфы — в валидацию, и метрика будет мерить
   запоминание текста. Валидация собирается из двух срезов в пропорции
   бенчмарка, где текст запроса встречается в train у 37% строк (разбиение по
   событиям дало бы 85.7%, только по тексту — 0%):

   * «новый текст» — все события отложенных `val_text_fraction` текстов;
   * «знакомый текст» — события текстов, оставшихся в обучении; у каждого
     такого текста минимум одно событие остаётся в train.

3. **Каталог** — объявления, среди которых идёт поиск: все уникальные
   `item_id`, прореженные до размера корпуса бенчмарка (189 212). Режим
   `uniform` — равномерная выборка, так устроен и сам корпус: часть позитивов
   в него не попадает (сейчас 54.6% позитивов val в каталоге). Режим
   `keep_positives` сохранил бы все позитивы val, но их доля в каталоге выросла
   бы вдвое и задача стала бы искусственно лёгкой.

4. **Валидация к оценке** (`val_eval`) — позитивы, попавшие в каталог. События
   без позитивов в каталоге оценивать нечем. События с числом позитивов больше
   `max_positives` отбрасываются: у них потолок recall@K ниже 100% арифметически.

Результат — `dataset/split/`: `train_pairs`, `val_pairs`, `val_eval`,
`catalog` и `config.json`. Колонка события (кортеж) в parquet не сохраняется,
вместо неё целочисленный `event_id`; сам ключ восстанавливается из колонок
`EVENT_COLS`.

Порядок вызовов генератора случайных чисел важен для воспроизводимости:
при том же `seed` разбиение получается тем же.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from paths import PROCESSED

EVENT_COLS = ["search_query", "search_location_id", "filter_category",
              "filter_subcategory", "filter_subject", "filter_online_booking"]
UNSEEN, SEEN = "новый текст", "знакомый текст"
CATALOG_MODES = ("uniform", "keep_positives", "full")


@dataclass
class SplitConfig:
    seed: int = 42
    dedup: bool = True               # схлопывать (событие, item_id)
    val_text_fraction: float = 0.10  # доля текстов запроса, целиком уходящих в val
    seen_text_share: float = 0.37    # доля val-событий со знакомым текстом — как в бенчмарке
    catalog_mode: str = "uniform"    # "uniform" | "keep_positives" | "full"
    catalog_size: int = 189_212      # размер корпуса бенчмарка
    top_k: int = 50
    max_positives: int = 50          # события с бóльшим числом позитивов не оцениваем

    def to_json(self) -> dict:
        # ключи в верхнем регистре — формат config.json, который читают другие модули
        return {k.upper(): v for k, v in asdict(self).items()}


@dataclass
class Split:
    train_pairs: pd.DataFrame
    val_pairs: pd.DataFrame
    val_eval: pd.DataFrame
    catalog: set
    n_items_total: int


def event_key(df: pd.DataFrame) -> list[tuple]:
    """Хешируемый ключ поискового события — те же поля, что в benchmark_queries."""
    return list(zip(
        df.search_query,
        df.search_location_id,
        df.filter_category.fillna(""),                          # NaN у 166 тыс. строк
        df.filter_subcategory.map(lambda x: tuple(sorted(x))),  # из parquet приходит ndarray
        df.filter_subject.map(lambda x: tuple(sorted(x))),
        df.filter_online_booking,
    ))


def load_pairs(dedup: bool = True) -> pd.DataFrame:
    """Пары «событие → объявление» из обработанного train."""
    pairs = pq.read_table(PROCESSED / "train.parquet",
                          columns=EVENT_COLS + ["item_id"]).to_pandas()
    pairs["event"] = event_key(pairs)
    if dedup:
        counts = pairs.groupby(["event", "item_id"], sort=False).size().rename("n_repeats")
        pairs = pairs.drop_duplicates(["event", "item_id"]).merge(counts, on=["event", "item_id"])
    else:
        pairs["n_repeats"] = 1
    return pairs


def split_by_text(pairs: pd.DataFrame, cfg: SplitConfig, rng: np.random.Generator):
    """Возвращает (train_pairs, val_pairs); у val_pairs колонка `slice`."""
    texts = pairs.search_query.unique()
    held_texts = set(rng.choice(texts, size=int(len(texts) * cfg.val_text_fraction),
                                replace=False))

    unseen = pairs[pairs.search_query.isin(held_texts)]
    rest = pairs[~pairs.search_query.isin(held_texts)]

    # сколько событий со знакомым текстом нужно добрать до доли бенчмарка
    n_unseen_events = unseen.event.nunique()
    n_seen_events = int(round(n_unseen_events * cfg.seen_text_share
                              / (1 - cfg.seen_text_share)))

    # у каждого текста одно событие оставляем в train, остальные — кандидаты в val
    rest_events = rest[["search_query", "event"]].drop_duplicates()
    shuffled = rest_events.sample(frac=1.0, random_state=cfg.seed)
    candidates = shuffled[shuffled.duplicated("search_query", keep="first")].event.to_numpy()
    seen_events = set(rng.choice(candidates, size=n_seen_events, replace=False))

    seen = rest[rest.event.isin(seen_events)]
    train_pairs = rest[~rest.event.isin(seen_events)]
    val_pairs = pd.concat([unseen.assign(slice=UNSEEN), seen.assign(slice=SEEN)],
                          ignore_index=True)

    train_texts = set(train_pairs.search_query)
    assert not (set(unseen.search_query) & train_texts), "текст из среза «новый» остался в train"
    assert set(seen.search_query) <= train_texts, "текст из среза «знакомый» отсутствует в train"
    assert not (set(train_pairs.event) & set(val_pairs.event)), "событие попало в обе части"
    return train_pairs, val_pairs


def sample_catalog(pairs: pd.DataFrame, val_pairs: pd.DataFrame, cfg: SplitConfig,
                   rng: np.random.Generator) -> set:
    all_items = np.array(sorted(set(pairs.item_id)))
    val_positive_items = set(val_pairs.item_id)
    if cfg.catalog_mode == "full" or cfg.catalog_size >= len(all_items):
        return set(all_items)
    if cfg.catalog_mode == "uniform":
        return set(rng.choice(all_items, size=cfg.catalog_size, replace=False))
    if cfg.catalog_mode == "keep_positives":
        others = np.array(sorted(set(all_items) - val_positive_items))
        fill = rng.choice(others, size=max(0, cfg.catalog_size - len(val_positive_items)),
                          replace=False)
        return val_positive_items | set(fill)
    raise ValueError(f"неизвестный catalog_mode: {cfg.catalog_mode}")


def select_eval(val_pairs: pd.DataFrame, catalog: set, cfg: SplitConfig) -> pd.DataFrame:
    """Позитивы в каталоге у событий, где их от 1 до max_positives."""
    in_catalog = val_pairs[val_pairs.item_id.isin(catalog)]
    per_event = in_catalog.groupby("event").size()
    keep = set(per_event[per_event <= cfg.max_positives].index)
    return in_catalog[in_catalog.event.isin(keep)]


def make_split(cfg: SplitConfig) -> Split:
    rng = np.random.default_rng(cfg.seed)
    pairs = load_pairs(cfg.dedup)
    train_pairs, val_pairs = split_by_text(pairs, cfg, rng)
    catalog = sample_catalog(pairs, val_pairs, cfg, rng)
    val_eval = select_eval(val_pairs, catalog, cfg)
    return Split(train_pairs, val_pairs, val_eval, catalog, pairs.item_id.nunique())


def save_split(split: Split, cfg: SplitConfig, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    codes, _ = pd.factorize(pd.concat([split.train_pairs.event, split.val_pairs.event]),
                            sort=False)
    n_train = len(split.train_pairs)
    train_pairs = split.train_pairs.assign(event_id=codes[:n_train])
    val_pairs = split.val_pairs.assign(event_id=codes[n_train:])
    val_eval = val_pairs[val_pairs.event.isin(set(split.val_eval.event))
                         & val_pairs.item_id.isin(split.catalog)]

    for name, df in {"train_pairs": train_pairs, "val_pairs": val_pairs,
                     "val_eval": val_eval}.items():
        df.drop(columns=["event"]).to_parquet(out / f"{name}.parquet", index=False)
    pd.DataFrame({"item_id": sorted(split.catalog)}).to_parquet(out / "catalog.parquet",
                                                                index=False)
    (out / "config.json").write_text(json.dumps(cfg.to_json(), ensure_ascii=False, indent=2),
                                     encoding="utf-8")


def report(split: Split, cfg: SplitConfig) -> pd.DataFrame:
    """Сводка по частям разбиения."""
    tr, val, ev = split.train_pairs, split.val_pairs, split.val_eval
    parts = {"train": tr, f"val: {UNSEEN}": val[val.slice == UNSEEN],
             f"val: {SEEN}": val[val.slice == SEEN], "val всего": val,
             "val к оценке": ev}
    return pd.DataFrame({
        "событий": [df.event.nunique() for df in parts.values()],
        "примеров": [len(df) for df in parts.values()],
        "текстов": [df.search_query.nunique() for df in parts.values()],
    }, index=list(parts))

"""Признаки пар «запрос — кандидат» для переранжировщика (docs/reranker_plan.md).

Группы признаков (префикс колонки):

* `sc_` — текст: скоры BM25, e5, RoSBERTa и их разрыв с лучшим кандидатом запроса;
  ранги ретриверов и RRF внутри гео и вне его (из кандидатов), позиция и источник;
* `geo_` — тот же город, запрос с регионом, расстояние до центра локации запроса,
  размер локации объявления;
* `flt_` — совпадение с каждым фильтром запроса (NaN, если фильтра нет);
* `it_` — объявление: рейтинг, отзывы, цена, микрокатегория, вид услуги, флаги,
  длины заголовка и описания;
* `st_` — статистики по парам train: частота текста, сколько раз объявление было
  позитивом для того же текста и вообще, доля микрокатегории среди позитивов
  текста, доля локации объявления среди позитивов локации запроса;
* `q_` — запрос: длина.

**Без утечки.** Статистики для обучающих событий считаются leave-one-event-out:
по всем парам train, кроме самого события. Так обучающее событие видит train
так же, как событие валидации: у знакомого текста есть другие события, у нового —
нет. Для валидации статистики — по train_pairs, для бенчмарка — по всем парам.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

CATEGORICAL = ["it_microcat", "it_category"]


# --- объявления ---------------------------------------------------------------


def item_features(items: pd.DataFrame, title_len: np.ndarray, desc_len: np.ndarray
                  ) -> pd.DataFrame:
    """Признаки объявлений корпуса; строка i — объявление с индексом i."""
    price = items.item_price.to_numpy(dtype=float)
    price_ok = (price > 0) & (price < 1e9)
    loc_size = items.item_location_id.map(items.item_location_id.value_counts())
    return pd.DataFrame({
        "it_rating": items.item_rating.to_numpy(dtype=np.float32),
        "it_reviews_log": np.log1p(items.item_rating_reviews_count.fillna(0)).astype(np.float32),
        "it_price_log": np.where(price_ok, np.log1p(np.where(price_ok, price, 0)), np.nan)
        .astype(np.float32),
        "it_microcat": items.item_microcat_id.to_numpy(dtype=np.int32),
        "it_category": category_code(items.item_category),
        "it_is_service": items.item_is_service.to_numpy(dtype=np.int8),
        "it_phone_hidden": items.item_is_phone_hidden.to_numpy(dtype=np.int8),
        "it_msg_forbidden": items.item_is_message_forbidden.to_numpy(dtype=np.int8),
        "it_title_len": np.asarray(title_len, dtype=np.int32),
        "it_desc_len_log": np.log1p(np.asarray(desc_len)).astype(np.float32),
        "geo_loc_size_log": np.log1p(loc_size.to_numpy()).astype(np.float32),
    })


def category_code(names: pd.Series) -> np.ndarray:
    """Код вида услуги из его названия — одинаковый в любом корпусе.

    (pd.factorize нумерует в порядке появления, и у каталога валидации и корпуса
    бенчмарка один номер означал бы разные виды услуг.)
    """
    return np.array([zlib.crc32(s.encode("utf-8")) % 1_000_003 for s in names.fillna("")],
                    dtype=np.int32)


def location_centroids(locations: pd.Series, lat: pd.Series, lon: pd.Series) -> pd.DataFrame:
    """Медианные координаты объявлений каждой локации (без точек (0, 0))."""
    df = pd.DataFrame({"loc": locations.to_numpy(), "lat": lat.to_numpy(), "lon": lon.to_numpy()})
    df = df[(df.lat != 0) | (df.lon != 0)].dropna()
    return df.groupby("loc")[["lat", "lon"]].median()


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(h))


# --- статистики по train --------------------------------------------------------


class TrainStats:
    """Счётчики по парам train: `event_id, search_query, search_location_id,
    item_id, item_microcat_id, item_location_id` (пара = позитив события)."""

    def __init__(self, pairs: pd.DataFrame):
        self.pairs = pairs
        p = pairs
        self.text_events = p.groupby("search_query").event_id.nunique().rename("text_events")
        self.text_pos = p.groupby("search_query").size().rename("text_pos")
        self.text_item = p.groupby(["search_query", "item_id"]).size().rename("text_item")
        self.text_mc = p.groupby(["search_query", "item_microcat_id"]).size().rename("text_mc")
        self.item_pop = p.groupby("item_id").size().rename("item_pop")
        self.loc_loc = p.groupby(["search_location_id", "item_location_id"]).size().rename("loc_loc")
        self.loc_pos = p.groupby("search_location_id").size().rename("loc_pos")
        # вклад каждого события — для leave-one-event-out
        self.own_pos = p.groupby("event_id").size().rename("own_pos")
        self.own_item = p.groupby(["event_id", "item_id"]).size().rename("own_item")
        self.own_mc = p.groupby(["event_id", "item_microcat_id"]).size().rename("own_mc")
        self.own_loc = p.groupby(["event_id", "item_location_id"]).size().rename("own_loc")

    def features(self, df: pd.DataFrame, leave_one_out: bool) -> pd.DataFrame:
        """`df`: qid, search_query, search_location_id, item_id, item_microcat_id,
        item_location_id. При leave_one_out qid — event_id из тех же пар."""
        x = df[["qid", "search_query", "search_location_id", "item_id", "item_microcat_id",
                "item_location_id"]].copy()

        def join(series, keys, own=None, own_keys=None):
            vals = x.join(series, on=keys)[series.name].fillna(0).to_numpy(dtype=np.float64)
            if leave_one_out and own is not None:
                vals -= x.join(own, on=own_keys)[own.name].fillna(0).to_numpy(dtype=np.float64)
            return vals

        in_pairs = x.qid.isin(self.own_pos.index).to_numpy() if leave_one_out else False
        text_events = join(self.text_events, "search_query") - np.where(in_pairs, 1, 0)
        text_pos = join(self.text_pos, "search_query", self.own_pos, "qid")
        text_item = join(self.text_item, ["search_query", "item_id"],
                         self.own_item, ["qid", "item_id"])
        text_mc = join(self.text_mc, ["search_query", "item_microcat_id"],
                       self.own_mc, ["qid", "item_microcat_id"])
        item_pop = join(self.item_pop, "item_id", self.own_item, ["qid", "item_id"])
        loc_loc = join(self.loc_loc, ["search_location_id", "item_location_id"],
                       self.own_loc, ["qid", "item_location_id"])
        loc_pos = join(self.loc_pos, "search_location_id", self.own_pos, "qid")

        with np.errstate(divide="ignore", invalid="ignore"):
            return pd.DataFrame({
                "st_text_freq_log": np.log1p(text_events).astype(np.float32),
                "st_text_item": text_item.astype(np.float32),
                "st_text_mc_share": np.where(text_pos > 0, text_mc / text_pos, np.nan)
                .astype(np.float32),
                "st_item_pop_log": np.log1p(item_pop).astype(np.float32),
                "st_loc_share": np.where(loc_pos > 0, loc_loc / loc_pos, np.nan).astype(np.float32),
                "st_loc_pos_log": np.log1p(loc_pos).astype(np.float32),
            }, index=df.index)


# --- фильтры ------------------------------------------------------------------


def filter_features(cand: pd.DataFrame, events: pd.DataFrame, items: pd.DataFrame
                    ) -> pd.DataFrame:
    """Совпадение кандидата с каждым фильтром запроса; NaN — фильтра нет."""
    category = items.item_category.fillna("").to_numpy()
    subcategory = items.item_subcategory.fillna("").to_numpy()
    subjects = items.item_subjects.map(set).to_numpy()
    booking = items.item_online_booking.to_numpy()

    # фильтры — свойства запроса: считаем по запросам и разворачиваем на кандидатов
    ev = events[["filter_category", "filter_subcategory", "filter_subject",
                 "filter_online_booking"]].copy()
    ev["sub_key"] = ev.filter_subcategory.map(lambda v: tuple(sorted(v)))
    ev["subj_key"] = ev.filter_subject.map(lambda v: tuple(sorted(v)))
    pos = pd.Series(np.arange(len(ev)), index=ev.index)
    q = pos.loc[cand.qid.to_numpy()].to_numpy()
    idx = cand.item_idx.to_numpy()

    out = {c: np.full(len(cand), np.nan, dtype=np.float32)
           for c in ("flt_category", "flt_subcategory", "flt_subject", "flt_booking")}
    f_cat = ev.filter_category.fillna("").to_numpy()[q]
    has = f_cat != ""
    out["flt_category"][has] = category[idx[has]] == f_cat[has]

    # тип и предмет: маска по корпусу на каждое уникальное значение фильтра
    for key_col, name, item_match in (
            ("sub_key", "flt_subcategory", lambda want: np.isin(subcategory, want)),
            ("subj_key", "flt_subject",
             lambda want: np.array([bool(s & set(want)) for s in subjects]))):
        keys = ev[key_col].to_numpy()[q]
        codes, uniques = pd.factorize(keys)
        for code, want in enumerate(uniques):
            if not want:
                continue
            rows = codes == code
            out[name][rows] = item_match(list(want))[idx[rows]]

    has = ev.filter_online_booking.to_numpy().astype(bool)[q]
    out["flt_booking"][has] = booking[idx[has]]
    n_filters = ((ev.filter_category.fillna("") != "").astype(int) + ev.sub_key.map(bool)
                 + ev.subj_key.map(bool) + ev.filter_online_booking.astype(int)).to_numpy()
    out["flt_n_filters"] = n_filters[q].astype(np.int8)
    return pd.DataFrame(out, index=cand.index)


# --- сборка -------------------------------------------------------------------


def score_gaps(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Разрыв скора с лучшим кандидатом того же запроса (0 у лучшего)."""
    return pd.DataFrame({f"{c}_gap": (df[c] - df.groupby("qid")[c].transform("max"))
                         .astype(np.float32) for c in columns}, index=df.index)

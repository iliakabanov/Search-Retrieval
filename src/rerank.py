"""Переранжировщик кандидатов: данные для обучения и оценка recall@K.

Признаки — `features.py`, кандидаты — `candidates.py`. Модель ставит скор каждой
паре «запрос — кандидат»; ответ — топ-K кандидатов запроса по скору.
Recall считается относительно всех позитивов события (`n_pos` из событий), а не
только найденных кандидатами: позитив, не попавший в кандидаты, — промах.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features import CATEGORICAL

#: Служебные колонки таблицы признаков — не признаки модели.
SERVICE = ["qid", "item_idx", "label", "fold", "val_part"]


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in SERVICE]


def drop_groups_without_positives(df: pd.DataFrame) -> pd.DataFrame:
    """Запросы, где среди кандидатов нет позитива, ранжированию ничего не дают."""
    has_pos = df.groupby("qid").label.transform("max") > 0
    return df[has_pos]


def group_sizes(df: pd.DataFrame) -> np.ndarray:
    """Размеры групп для ранжирующей модели; df отсортирован по qid."""
    return df.groupby("qid", sort=False).size().to_numpy()


def top_k_recall(df: pd.DataFrame, score: np.ndarray, n_pos: pd.Series, k: int) -> pd.Series:
    """Recall@k по запросам: доля позитивов события среди k кандидатов с лучшим скором."""
    d = pd.DataFrame({"qid": df.qid.to_numpy(), "label": df.label.to_numpy(), "score": score})
    d["rank"] = d.groupby("qid").score.rank(method="first", ascending=False)
    found = d[(d["rank"] <= k) & (d.label == 1)].groupby("qid").size()
    return (found.reindex(n_pos.index, fill_value=0) / n_pos).rename("recall")


def report(val: pd.DataFrame, scores: dict[str, np.ndarray], val_events: pd.DataFrame,
           baseline: pd.Series, k: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recall@k на val-tune / val-test (+ город / регион) для нескольких скоров.

    `scores` — {название модели: скор каждой строки val}; `baseline` — recall нынешнего
    метода по событиям. Возвращает таблицу и recall по событиям.
    """
    per_event = val_events[["val_part", "is_region", "slice"]].assign(
        **{"RRF (сейчас)": baseline},
        **{name: top_k_recall(val, s, val_events.n_pos, k) for name, s in scores.items()},
        **{"потолок кандидатов": top_k_recall(val, np.zeros(len(val)), val_events.n_pos, 10 ** 6)})
    columns = ["RRF (сейчас)", *scores, "потолок кандидатов"]
    rows = {}
    for part in ("tune", "test"):
        p = per_event[per_event.val_part == part]
        rows[f"val-{part}"] = p[columns].mean()
        for grp, mask in (("город", ~p.is_region.astype(bool)), ("регион", p.is_region.astype(bool))):
            rows[f"val-{part}, {grp}"] = p[mask][columns].mean()
    table = pd.DataFrame(rows).T
    for name in scores:
        table[f"прирост {name}, п.п."] = (table[name] - table["RRF (сейчас)"]) * 100
    return table, per_event


def format_report(table: pd.DataFrame) -> str:
    return table.to_string(formatters={c: ("{:+.2f}".format if c.startswith("прирост")
                                           else "{:.2%}".format) for c in table.columns})


def prepare_categorical(df: pd.DataFrame) -> pd.DataFrame:
    for c in CATEGORICAL:
        if c in df:
            df[c] = df[c].astype("int32")
    return df

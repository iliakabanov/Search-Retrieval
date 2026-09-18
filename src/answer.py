"""Файл ответа answer.csv и проверка требований к нему.

Формат: CSV через запятую, UTF-8, ровно две колонки:

* `query_id` — идентификатор из benchmark_queries (16 символов, регистр важен);
* `answer` — до 50 item_id через пробел, каждый — 16 символов 0-9a-f, как в
  benchmark_items.

Строка на каждый query_id без пропусков, лишних строк и повторов; внутри строки
item_id не повторяются и все существуют в benchmark_items. Метрика — Recall@50,
порядок объявлений в ответе на неё не влияет.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

MAX_ITEMS = 50
QUERY_ID = re.compile(r"^.{16}$")
ITEM_ID = re.compile(r"^[0-9a-f]{16}$")


def make_answer(predictions: dict[str, list[str]], query_ids) -> pd.DataFrame:
    """Строка на каждый query_id; запросы без предсказаний получают пустой ответ."""
    query_ids = list(query_ids)
    return pd.DataFrame({
        "query_id": query_ids,
        "answer": [" ".join(predictions.get(q, [])) for q in query_ids],
    })


def check_answer(answer: pd.DataFrame, query_ids, item_ids) -> list[str]:
    """Нарушения требований к ответу; пустой список — всё в порядке."""
    problems = []
    if list(answer.columns) != ["query_id", "answer"]:
        problems.append(f"колонки {list(answer.columns)}, нужны ровно ['query_id', 'answer']")
        return problems

    expected = set(query_ids)
    got = answer.query_id.tolist()
    if len(got) != len(set(got)):
        problems.append(f"повторы query_id: {len(got) - len(set(got))}")
    if missing := expected - set(got):
        problems.append(f"нет строк для {len(missing)} query_id")
    if extra := set(got) - expected:
        problems.append(f"лишние query_id: {len(extra)}")
    if bad := [q for q in got if not QUERY_ID.match(str(q))]:
        problems.append(f"query_id не из 16 символов: {len(bad)}")

    known = set(item_ids)
    too_long = dup = unknown = bad_format = 0
    for text in answer.answer.fillna(""):
        items = text.split()
        too_long += len(items) > MAX_ITEMS
        dup += len(items) != len(set(items))
        unknown += sum(i not in known for i in items)
        bad_format += sum(not ITEM_ID.match(i) for i in items)
    for n, what in ((too_long, f"строк с числом item_id больше {MAX_ITEMS}"),
                    (dup, "строк с повторами item_id"),
                    (unknown, "item_id, которых нет в benchmark_items"),
                    (bad_format, "item_id не из 16 символов 0-9a-f")):
        if n:
            problems.append(f"{what}: {n}")
    return problems


def save_answer(answer: pd.DataFrame, path: Path) -> None:
    # \n явно: на Windows pandas по умолчанию пишет \r\n
    answer.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")

"""Текст документа объявления — общий вход для всех ретриверов.

Документ — заголовок плюс описание, очищенные от эмодзи, ссылок, почт и
телефонов. Один и тот же текст индексирует BM25 и кодируют dense-модели, так
что ретриверы сравниваются на одинаковом входе. Заголовок идёт первым: он
короткий и точный, а при обрезке по длине (dense-модели видят первые 512
токенов) первым теряется хвост описания.

Если чистка меняется, кэш эмбеддингов в `dataset/embeddings/` нужно
пересчитать: он хранит эмбеддинги старых текстов.
"""

from __future__ import annotations

import re

import pandas as pd

EMOJI = re.compile("[\U0001F000-\U0001FAFF☀-➿️⬀-⯿]")
JUNK = re.compile(r"https?://\S+|www\.\S+|\S+@\S+\.\w+|&[a-z]{2,6};")
PHONE = re.compile(r"(?:\+7|8)[\s\-(]*\d{3}[\s\-)]*\d{3}[\s\-]*\d{2}[\s\-]*\d{2}")


def clean_text(s: str) -> str:
    """Убирает эмодзи, ссылки, почты, телефоны и схлопывает пробелы."""
    s = EMOJI.sub(" ", s)
    s = JUNK.sub(" ", s)
    s = PHONE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def doc_texts(items: pd.DataFrame) -> list[str]:
    """Документ объявления: заголовок + описание, очищенные."""
    raw = items.item_title_raw.fillna("") + " " + items.item_description_raw.fillna("")
    return [clean_text(d) for d in raw]

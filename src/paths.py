"""Пути проекта."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATASET = ROOT / "dataset"              # сырые parquet-файлы
PROCESSED = DATASET / "processed"       # результат run/preprocess.py
SPLIT = DATASET / "split"               # результат run/make_split.py
EMBEDDINGS = DATASET / "embeddings"     # кэш эмбеддингов каталога (run/encode_catalog.py)
RESULTS = ROOT / "results"              # таблицы метрик

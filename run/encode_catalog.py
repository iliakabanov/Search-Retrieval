"""Кодирует корпус dense-моделями заранее и кладёт эмбеддинги в кэш (см. src/dense.py).

Необязательный шаг: run/evaluate_retrievers.py и run/retrieve_benchmark.py
досчитают недостающее сами, но кодирование занимает около часа на модель, и его
удобно запустить отдельно. Прерванный запуск продолжается с последней готовой части.

Корпус: `catalog` — каталог валидации из dataset/split (для оценки),
`benchmark` — benchmark_items (для ретрива по запросам бенчмарка).

    python run/encode_catalog.py
    python run/encode_catalog.py --models e5-large
    python run/encode_catalog.py --corpus benchmark
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import dense
from data import TEXT_ITEM_COLUMNS, load_benchmark_items, load_catalog
from logs import log, step
from texts import doc_texts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=list(dense.MODELS),
                        choices=list(dense.MODELS))
    parser.add_argument("--corpus", default=dense.CATALOG, choices=[dense.CATALOG, dense.BENCHMARK],
                        help="catalog — каталог валидации, benchmark — benchmark_items")
    args = parser.parse_args(argv)

    log(f"кодирование корпуса {args.corpus}: {', '.join(args.models)}")
    with step(f"читаем корпус {args.corpus}"):
        load = load_catalog if args.corpus == dense.CATALOG else load_benchmark_items
        items = load(TEXT_ITEM_COLUMNS)
    with step(f"собираем и чистим тексты {len(items):,} документов"):
        docs = doc_texts(items)

    for i, name in enumerate(args.models, start=1):
        with step(f"{name} ({i}/{len(args.models)}): эмбеддинги корпуса "
                  f"(готовые части берутся из кэша {dense.cache_dir(name, args.corpus)})"):
            emb = dense.catalog_embeddings(name, items.item_id, docs, corpus=args.corpus)
        print(f"  {name}: {emb.shape[0]:,} × {emb.shape[1]}")

    log("готово")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

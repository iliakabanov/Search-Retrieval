"""Сравнение ретриверов на валидации: recall@K при четырёх вариантах отсечения.

Ретриверы: BM25 и dense-модели из src/dense.py (без дообучения), плюс их
слияние RRF. Для ориентира в таблице есть потолок (доля позитивов, переживших
отсечение) и случайный порядок внутри пула. Подробности — в src/evaluate.py.

Пятый столбец «гео + добивка» — то, как собирается ответ на бенчмарке
(run/retrieve_benchmark.py): топ «только гео», добитый до K лучшими из «только
фильтры». Его потолок — доля позитивов, попавших хотя бы в один из двух пулов
(верхняя оценка), случайного порядка для него нет.

Печатает таблицы и сохраняет их в results/<имя>/ вместе с recall по каждому
событию (per_event.parquet) — для разбора ошибок без перезапуска.

Нужно окружение с torch и sentence-transformers (torch_env), если в списке есть
dense-модели. Эмбеддинги каталога берутся из кэша; недостающие досчитываются
(около часа на модель, см. run/encode_catalog.py).

    python run/evaluate_retrievers.py
    python run/evaluate_retrievers.py --retrievers BM25 e5-large
    python run/evaluate_retrievers.py --retrievers BM25 --no-stemming --name bm25_no_stem
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import pandas as pd
from bm25 import BM25
from data import (FILTER_ITEM_COLUMNS, TEXT_ITEM_COLUMNS, load_catalog, load_geo_pairs,
                  load_item_locations, load_split_config, load_val_eval)
from evaluate import (build_events, complementarity, evaluate, group_table, hit_rate_table,
                      per_event_recall, recall_table, slice_table)
from geo import region_map
from logs import log, step
from paths import RESULTS
from pools import FILLED, CandidatePools, cutoff_stats
from texts import doc_texts

RETRIEVER_NAMES = ["BM25", "e5-large", "RoSBERTa"]


def pct(df: pd.DataFrame, digits: int = 2) -> str:
    return df.to_string(float_format=lambda x: f"{x:.{digits}%}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--retrievers", nargs="+", default=RETRIEVER_NAMES,
                        choices=RETRIEVER_NAMES)
    parser.add_argument("--no-rrf", action="store_true", help="не считать слияние RRF")
    parser.add_argument("--rrf-k", type=int, default=60,
                        help="вклад документа в RRF = 1 / (rrf_k + ранг)")
    parser.add_argument("--rrf-depth", type=int, default=200,
                        help="сколько лучших из пула каждого ретривера сливать")
    parser.add_argument("--region-cover", type=float, default=0.95,
                        help="гео региона: его локации, покрывающие эту долю позитивов "
                             "в train_pairs (0 — пул региона пустой)")
    parser.add_argument("--k1", type=float, default=1.2, help="BM25: k1")
    parser.add_argument("--b", type=float, default=0.75, help="BM25: b")
    parser.add_argument("--no-stemming", action="store_true", help="BM25 без стемминга")
    parser.add_argument("--name", default="retrievers", help="папка результатов в results/")
    args = parser.parse_args(argv)

    t_start = time.time()
    config = load_split_config()
    top_k = config["TOP_K"]
    pd.set_option("display.width", 220)
    log(f"оценка ретриверов: {', '.join(args.retrievers)}"
        f"{'' if args.no_rrf or len(args.retrievers) < 2 else ' + RRF'}, recall@{top_k}")

    # данные
    with step("читаем валидацию и каталог"):
        val = load_val_eval()
        items = load_catalog(TEXT_ITEM_COLUMNS + FILTER_ITEM_COLUMNS)
    with step(f"собираем и чистим тексты {len(items):,} документов"):
        docs = doc_texts(items)
        items = items.drop(columns=TEXT_ITEM_COLUMNS)   # тексты уже в docs
        gc.collect()
    with step(f"строим гео регионов по train_pairs (покрытие {args.region_cover:.0%})"):
        regions = region_map(load_geo_pairs(train_pairs_only=True),
                             set(load_item_locations()), args.region_cover)
    print(f"  регионов: {len(regions):,}")
    with step("готовим события валидации и маски фильтров"):
        pools = CandidatePools(items, regions)
        events = build_events(val, pools, items.item_id)
        events["гео запроса"] = (val.groupby("event_id").search_location_id.first()
                                 .reindex(events.index).isin(regions)
                                 .map({True: "регион", False: "город"}))
    print(f"  каталог: {len(items):,} объявлений | валидация: {len(events):,} событий, "
          f"{len(val):,} позитивов | уникальных текстов запроса: {events['query'].nunique():,}")

    with step("считаем потолок и случайный порядок для 4 вариантов отсечения"):
        cutoffs = cutoff_stats(events, pools, top_k)

    # ретриверы
    retrievers = []
    for name in args.retrievers:
        if name == "BM25":
            with step("BM25: строим индекс по каталогу"):
                r = BM25(docs, k1=args.k1, b=args.b, stemming=not args.no_stemming)
            print(f"  {r.n_words:,} слов -> {r.n_terms:,} термов, "
                  f"средняя длина документа {r.avg_len:.0f} токенов")
        else:
            from dense import DenseRetriever
            with step(f"{name}: загружаем эмбеддинги каталога (недостающие досчитываются)"):
                r = DenseRetriever(name, items.item_id, docs)
        retrievers.append(r)
    del docs
    gc.collect()

    hits = evaluate(events, pools, retrievers, top_k, fuse=not args.no_rrf,
                    rrf_k=args.rrf_k, rrf_depth=args.rrf_depth)
    with step("собираем таблицы"):
        recall = per_event_recall(hits, events, cutoffs)

    # таблицы
    names = [r.name for r in retrievers]
    tables = {
        f"recall@{top_k}": recall_table(recall),
        f"hit-rate@{top_k}": hit_rate_table(recall),
        "recall по срезам": slice_table(recall, events),
        "recall по гео запроса (город / регион вместо города)":
            group_table(recall, events["гео запроса"], ["только гео", FILLED]),
    }
    if len(names) > 1:
        for v in ("без отсечения", "только фильтры"):
            tables[f"дополнительность, {v}: нашёл строка, не нашёл столбец"] = \
                complementarity(recall, names, v)

    for title, table in tables.items():
        print(f"\n{title}\n{pct(table, 1 if title.startswith('дополн') else 2)}")

    # сохранение
    out = RESULTS / args.name
    print()
    with step(f"сохраняем таблицы и recall по событиям в {out}"):
        out.mkdir(parents=True, exist_ok=True)
        files = ["recall", "hit_rate", "by_slice", "by_geo"] + [f"complementarity_{i}"
                                                                 for i in range(2)]
        for fname, table in zip(files, tables.values()):
            table.to_csv(out / f"{fname}.csv", encoding="utf-8-sig", float_format="%.6f")
        per_event = recall.copy()
        per_event.columns = [f"{n} | {v}" for n, v in per_event.columns]
        per_event.join(events[["query", "slice", "гео запроса", "n_pos"]]).to_parquet(
            out / "per_event.parquet")
        run_config = {"split": config, **{k: v for k, v in vars(args).items()}}
        (out / "config.json").write_text(json.dumps(run_config, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    log(f"готово за {(time.time() - t_start) / 60:.1f} мин")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

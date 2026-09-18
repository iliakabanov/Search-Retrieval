"""Ретрив по бенчмарку: до 50 объявлений из benchmark_items на каждый запрос
benchmark_queries — тем же алгоритмом, что оценивается на валидации, — и файл
ответа answer.csv.

Алгоритм (src/retrieve.py): отсечение кандидатов по варианту `--variant`,
ранжирование каждым ретривером (BM25, e5-large, RoSBERTa) и слияние RRF. По
умолчанию — лучшая конфигурация на валидации: RRF всех трёх ретриверов внутри
города запроса («только гео», recall@50 78.1% на валидации).

**Добивка до 50.** При отсечении по городу у части запросов кандидатов меньше 50
(город, где мало объявлений, или ни одного). Метрика — Recall@50, порядок в
ответе на неё не влияет, поэтому свободные места заполняются лучшими кандидатами
запасного варианта отсечения `--fill` (по умолчанию «только фильтры»: лучший
вариант без гео на валидации), которых ещё нет в ответе. Recall ни одного
запроса от этого не падает. Отключить: `--no-fill`.

Пишет в results/<имя>/:

| Файл | Что в нём |
|---|---|
| `answer.csv` | ответ в формате задания: query_id, answer (item_id через пробел) |
| `predictions.parquet` | query_id, rank, item_id, source (основной вариант / добивка) |
| `all_retrievers.parquet` | топ каждого ретривера и RRF по каждому варианту отсечения |
| `config.json` | параметры запуска |

answer.csv проверяется на все требования задания (src/answer.py); при нарушении
скрипт завершается с ошибкой.

Эмбеддинги корпуса бенчмарка кэшируются отдельно от каталога валидации
(dataset/embeddings/benchmark/); первый запуск с dense-моделями кодирует корпус
(около часа на модель, можно заранее: run/encode_catalog.py --corpus benchmark).

    python run/retrieve_benchmark.py
    python run/retrieve_benchmark.py --retrievers BM25 --name benchmark_bm25
    python run/retrieve_benchmark.py --variant "фильтры + гео" --fill "только гео"
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np
import pandas as pd
from answer import check_answer, make_answer, save_answer
from bm25 import BM25
from data import (FILTER_ITEM_COLUMNS, TEXT_ITEM_COLUMNS, load_benchmark_items,
                  load_benchmark_queries)
from logs import log, step
from paths import RESULTS
from pools import VARIANTS, CandidatePools
from retrieve import FUSED, build_queries, fill_top, search
from texts import doc_texts

RETRIEVER_NAMES = ["BM25", "e5-large", "RoSBERTa"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--retrievers", nargs="+", default=RETRIEVER_NAMES,
                        choices=RETRIEVER_NAMES)
    parser.add_argument("--variant", default="только гео", choices=VARIANTS,
                        help="основное отсечение кандидатов (по умолчанию: только гео)")
    parser.add_argument("--fill", default="только фильтры", choices=VARIANTS,
                        help="чем добивать ответ до --top-k (по умолчанию: только фильтры)")
    parser.add_argument("--no-fill", action="store_true", help="не добивать ответ до --top-k")
    parser.add_argument("--no-rrf", action="store_true",
                        help="не сливать ретриверы; итоговый — первый из --retrievers")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--rrf-k", type=int, default=60,
                        help="вклад документа в RRF = 1 / (rrf_k + ранг)")
    parser.add_argument("--rrf-depth", type=int, default=200,
                        help="сколько лучших из пула каждого ретривера сливать")
    parser.add_argument("--k1", type=float, default=1.2, help="BM25: k1")
    parser.add_argument("--b", type=float, default=0.75, help="BM25: b")
    parser.add_argument("--no-stemming", action="store_true", help="BM25 без стемминга")
    parser.add_argument("--name", default="benchmark", help="папка результатов в results/")
    args = parser.parse_args(argv)

    t_start = time.time()
    fuse = not args.no_rrf and len(args.retrievers) > 1
    final = FUSED if fuse else args.retrievers[0]
    fill = None if args.no_fill or args.fill == args.variant else args.fill
    variants = [args.variant] + ([fill] if fill else [])
    log(f"ретрив по бенчмарку: {', '.join(args.retrievers)}{' + RRF' if fuse else ''}, "
        f"отсечение «{args.variant}»{f', добивка из «{fill}»' if fill else ''}, "
        f"топ-{args.top_k}, итоговый ответ — {final}")

    # данные
    with step("читаем benchmark_queries и benchmark_items"):
        queries_raw = load_benchmark_queries()
        items = load_benchmark_items(TEXT_ITEM_COLUMNS + FILTER_ITEM_COLUMNS)
    with step(f"собираем и чистим тексты {len(items):,} документов"):
        docs = doc_texts(items)
        items = items.drop(columns=TEXT_ITEM_COLUMNS)   # тексты уже в docs
        gc.collect()
    with step("готовим запросы и маски фильтров"):
        pools = CandidatePools(items)
        queries = build_queries(queries_raw, pools, "query_id")
    print(f"  корпус: {len(items):,} объявлений | запросов: {len(queries):,} | "
          f"уникальных текстов: {queries['query'].nunique():,} | "
          f"город запроса отсутствует в корпусе: {(queries.loc_code < 0).sum():,}")

    # ретриверы
    retrievers = []
    for name in args.retrievers:
        if name == "BM25":
            with step("BM25: строим индекс по корпусу"):
                r = BM25(docs, k1=args.k1, b=args.b, stemming=not args.no_stemming)
            print(f"  {r.n_words:,} слов -> {r.n_terms:,} термов, "
                  f"средняя длина документа {r.avg_len:.0f} токенов")
        else:
            from dense import BENCHMARK, DenseRetriever
            with step(f"{name}: загружаем эмбеддинги корпуса (недостающие досчитываются)"):
                r = DenseRetriever(name, items.item_id, docs, corpus=BENCHMARK)
        retrievers.append(r)
    del docs
    gc.collect()

    # поиск
    item_ids = items.item_id.to_numpy()
    tops: dict[tuple[str, str], dict[str, np.ndarray]] = {}   # (query_id, вариант) -> топы
    for query_id, variant, result in search(queries, pools, retrievers, args.top_k, variants,
                                            fuse, args.rrf_k, args.rrf_depth):
        tops[(query_id, variant)] = result

    with step("собираем ответ" + (f" и добиваем до {args.top_k}" if fill else "")):
        rows = []
        for query_id in queries.index:
            main_top = item_ids[tops[(query_id, args.variant)][final]].tolist()
            ranked = (fill_top(main_top, item_ids[tops[(query_id, fill)][final]], args.top_k)
                      if fill else main_top)
            rows.append(pd.DataFrame({
                "query_id": query_id, "rank": np.arange(1, len(ranked) + 1), "item_id": ranked,
                "source": [args.variant] * len(main_top) + [fill] * (len(ranked) - len(main_top))}))
        predictions = pd.concat(rows, ignore_index=True)
        all_results = pd.concat([
            pd.DataFrame({"query_id": qid, "variant": v, "retriever": n,
                          "rank": np.arange(1, len(top) + 1), "item_id": item_ids[top]})
            for (qid, v), result in tops.items() for n, top in result.items()],
            ignore_index=True)

    with step("проверяем answer.csv на требования задания"):
        answer = make_answer(predictions.groupby("query_id", sort=False).item_id.apply(list)
                             .to_dict(), queries_raw.query_id)
        problems = check_answer(answer, queries_raw.query_id, items.item_id)
    for p in problems:
        log(f"ОШИБКА в ответе: {p}")
    if problems:
        return 1

    # сводка
    n_main = predictions[predictions.source == args.variant].groupby("query_id").size() \
        .reindex(queries.index, fill_value=0)
    n_total = predictions.groupby("query_id").size().reindex(queries.index, fill_value=0)
    print(f"\nитоговый ретривер: {final}, запросов: {len(queries):,}")
    print(f"  «{args.variant}»: полный топ-{args.top_k} у {(n_main == args.top_k).sum():,}, "
          f"короче у {((n_main > 0) & (n_main < args.top_k)).sum():,}, "
          f"пусто у {(n_main == 0).sum():,}")
    if fill:
        print(f"  добито из «{fill}»: {(predictions.source == fill).sum():,} объявлений "
              f"у {(n_total > n_main).sum():,} запросов")
    print(f"  в ответе: полный топ-{args.top_k} у {(n_total == args.top_k).sum():,}, "
          f"пусто у {(n_total == 0).sum():,}; всего {len(predictions):,} объявлений")

    # сохранение
    out = RESULTS / args.name
    print()
    with step(f"сохраняем ответ в {out}"):
        out.mkdir(parents=True, exist_ok=True)
        save_answer(answer, out / "answer.csv")
        predictions.to_parquet(out / "predictions.parquet", index=False)
        all_results.to_parquet(out / "all_retrievers.parquet", index=False)
        (out / "config.json").write_text(
            json.dumps({**vars(args), "final": final, "fill_used": fill},
                       ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"answer.csv: {out / 'answer.csv'}")
    log(f"готово за {(time.time() - t_start) / 60:.1f} мин")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

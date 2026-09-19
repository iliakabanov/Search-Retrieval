"""Кандидаты для переранжировщика (src/candidates.py, docs/reranker_plan.md).

Места в списке: `--k-geo` из гео запроса (город или локации региона), `--k-nb`
из соседних локаций города (центры не дальше `--nb-radius` км), остальные — вне
гео по фильтрам. По умолчанию 330 / 85 / 85 при K = 500, RRF сливает по 500
лучших от каждого ретривера. Потолок кандидатов на валидации — 97.6% (было 95.1%
при K = 300 без соседей и глубине RRF 300).

Части:

* `train` — события train_pairs с позитивом в каталоге (~173 тыс.). Делятся на
  фолды по тексту запроса; гео регионов для фолда строится по остальным фолдам,
  чтобы позитивы события не попадали в его же гео. Негативы прореживаются.
* `val` — события val_eval, все кандидаты; половина текстов — `tune` (ранняя
  остановка, гиперпараметры), половина — `test` (итоговое сравнение).
* `benchmark` — запросы бенчмарка по корпусу benchmark_items, гео регионов по всему train.

Пишет в dataset/rerank/: `<часть>_candidates.parquet` (пары запрос — кандидат) и
`<часть>_events.parquet` (запросы). Нужен torch_env: dense-ретриверы на GPU.

    python run/build_candidates.py --parts train val
    python run/build_candidates.py --parts benchmark
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from bm25 import BM25
from candidates import generate
from data import (FILTER_ITEM_COLUMNS, TEXT_ITEM_COLUMNS, load_benchmark_items,
                  load_benchmark_queries, load_catalog, load_geo_pairs, load_item_locations,
                  load_val_eval)
from evaluate import build_events
from geo import corpus_centroids, neighbor_map, region_map
from logs import log, step
from paths import RERANK, SPLIT
from pools import CandidatePools
from retrieve import build_queries
from texts import doc_texts

GEO_COLUMNS = ["item_latitude", "item_longitude"]
EVENT_META = ["search_query", "search_location_id", "filter_category", "filter_subcategory",
              "filter_subject", "filter_online_booking"]


def text_bucket(texts: pd.Series, n: int) -> np.ndarray:
    """Стабильный номер корзины по тексту запроса (фолды, tune/test)."""
    return np.array([zlib.crc32(t.encode("utf-8")) % n for t in texts])


def build_retrievers(items: pd.DataFrame, docs: list[str], corpus: str) -> list:
    from dense import DenseRetriever
    retrievers = []
    with step("BM25: строим индекс"):
        retrievers.append(BM25(docs))
    for name in ("e5-large", "RoSBERTa"):
        with step(f"{name}: загружаем эмбеддинги корпуса"):
            retrievers.append(DenseRetriever(name, items.item_id, docs, corpus=corpus))
    return retrievers


class Writer:
    """Пишет порции кандидатов в один parquet."""

    def __init__(self, path: Path):
        self.path, self.writer, self.rows = path, None, 0

    def write(self, df: pd.DataFrame) -> None:
        table = pa.Table.from_pandas(df, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema, compression="zstd")
        self.writer.write_table(table)
        self.rows += len(df)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def report(cands_path: Path, events: pd.DataFrame, k: int) -> None:
    c = pq.read_table(cands_path, columns=["qid", "label", "cand_rank"]).to_pandas()
    found = c[c.label == 1].groupby("qid").size().reindex(events.index, fill_value=0)
    top50 = c[(c.label == 1) & (c.cand_rank < 50)].groupby("qid").size() \
        .reindex(events.index, fill_value=0)
    n_pos = events.n_pos
    print(f"  позитивов среди {k} кандидатов (recall кандидатов): {(found / n_pos).mean():.1%}; "
          f"в первых 50 списка: {(top50 / n_pos).mean():.1%}; строк: {len(c):,}")


def run_val(items, docs, retrievers, args) -> None:
    val = load_val_eval()
    if args.max_val_events:
        keep = np.random.default_rng(0).choice(val.event_id.unique(), args.max_val_events,
                                               replace=False)
        val = val[val.event_id.isin(set(keep))]
    with step("гео регионов по train_pairs"):
        regions = region_map(load_geo_pairs(True), set(load_item_locations()), args.region_cover)
    pools = CandidatePools(items, regions, args.neighbors)
    events = build_events(val, pools, items.item_id)
    meta = val.groupby("event_id")[EVENT_META].first().reindex(events.index)
    events = events.join(meta.drop(columns="search_query"))
    events["is_region"] = events.search_location_id.isin(regions)
    events["val_part"] = np.where(text_bucket(events["query"], 2) == 0, "tune", "test")
    print(f"  событий: {len(events):,} (tune {int((events.val_part == 'tune').sum()):,}, "
          f"test {int((events.val_part == 'test').sum()):,})")

    out = Writer(args.out_dir / "val_candidates.parquet")
    for df in generate(events, pools, retrievers, args.k, args.quota, args.nb_quota,
                       gold=events.positives.to_dict(), rrf_depth=args.rrf_depth):
        out.write(df)
    out.close()
    save_events(events, "val", args.out_dir)
    report(out.path, events, args.k)


def run_train(items, docs, retrievers, args) -> None:
    tp = pq.read_table(SPLIT / "train_pairs.parquet").to_pandas()
    in_cat = tp[tp.item_id.isin(set(items.item_id))]
    per_event = in_cat.groupby("event_id").size()
    keep = per_event[per_event <= 50].index
    if args.max_train_events and len(keep) > args.max_train_events:
        keep = np.random.default_rng(0).choice(keep, args.max_train_events, replace=False)
    pairs = in_cat[in_cat.event_id.isin(set(keep))].assign(slice="train")
    pairs["fold"] = text_bucket(pairs.search_query, args.folds)
    geo_pairs_all = load_geo_pairs(True)
    geo_pairs_all["fold"] = text_bucket(
        pq.read_table(SPLIT / "train_pairs.parquet", columns=["search_query"]).column(0)
        .to_pandas(), args.folds)
    item_locs = set(load_item_locations())
    print(f"  событий: {pairs.event_id.nunique():,}, текстов: {pairs.search_query.nunique():,}, "
          f"фолдов: {args.folds}")

    out = Writer(args.out_dir / "train_candidates.parquet")
    all_events = []
    for fold in range(args.folds):
        log(f"фолд {fold + 1}/{args.folds}")
        with step("гео регионов по остальным фолдам"):
            regions = region_map(geo_pairs_all[geo_pairs_all.fold != fold],
                                 item_locs, args.region_cover)
        pools = CandidatePools(items, regions, args.neighbors)
        fold_pairs = pairs[pairs.fold == fold]
        events = build_events(fold_pairs, pools, items.item_id)
        meta = fold_pairs.groupby("event_id")[EVENT_META].first().reindex(events.index)
        events = events.join(meta.drop(columns="search_query"))
        events["is_region"] = events.search_location_id.isin(regions)
        events["fold"] = fold
        for df in generate(events, pools, retrievers, args.k, args.quota, args.nb_quota,
                           gold=events.positives.to_dict(), neg_top=args.neg_top,
                           neg_random=args.neg_random, seed=fold, rrf_depth=args.rrf_depth):
            out.write(df)
        all_events.append(events)
        gc.collect()
    out.close()
    events = pd.concat(all_events)
    save_events(events, "train", args.out_dir)
    report(out.path, events, args.k)


def run_benchmark(args) -> None:
    from dense import BENCHMARK
    with step("читаем корпус бенчмарка и тексты"):
        queries_raw = load_benchmark_queries()
        items = load_benchmark_items(TEXT_ITEM_COLUMNS + FILTER_ITEM_COLUMNS + GEO_COLUMNS)
        docs = doc_texts(items)
        items = items.drop(columns=TEXT_ITEM_COLUMNS)
    retrievers = build_retrievers(items, docs, BENCHMARK)
    del docs
    gc.collect()
    with step("гео регионов по всему train"):
        geo_pairs = load_geo_pairs(False)
        regions = region_map(geo_pairs, set(geo_pairs.item_location_id), args.region_cover)
    args.neighbors = neighbor_map(corpus_centroids(items), args.nb_radius)
    pools = CandidatePools(items, regions, args.neighbors)
    queries = build_queries(queries_raw, pools, "query_id")
    meta = queries_raw.set_index("query_id")[EVENT_META].reindex(queries.index)
    queries = queries.join(meta.drop(columns="search_query"))
    queries["is_region"] = queries.search_location_id.isin(regions)

    out = Writer(args.out_dir / "benchmark_candidates.parquet")
    for df in generate(queries, pools, retrievers, args.k, args.quota, args.nb_quota, rrf_depth=args.rrf_depth):
        out.write(df)
    out.close()
    save_events(queries, "benchmark", args.out_dir)
    print(f"  запросов: {len(queries):,}, строк: {out.rows:,}")


def save_events(events: pd.DataFrame, part: str, out_dir: Path) -> None:
    df = events.drop(columns=["fkey", "geo"], errors="ignore").copy()
    if "positives" in df:
        df["positives"] = df.positives.map(lambda s: sorted(int(x) for x in s))
    df.index.name = "qid"
    df.reset_index().to_parquet(out_dir / f"{part}_events.parquet", index=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parts", nargs="+", default=["train", "val"],
                        choices=["train", "val", "benchmark"])
    parser.add_argument("--k", type=int, default=500, help="кандидатов на запрос")
    parser.add_argument("--k-geo", type=int, default=330, help="мест из гео запроса")
    parser.add_argument("--k-nb", type=int, default=85,
                        help="мест из соседних локаций города (0 — без соседей)")
    parser.add_argument("--nb-radius", type=float, default=50.0, help="радиус соседей, км")
    parser.add_argument("--rrf-depth", type=int, default=500,
                        help="сколько лучших от каждого ретривера сливает RRF")
    parser.add_argument("--region-cover", type=float, default=0.95)
    parser.add_argument("--folds", type=int, default=5, help="фолдов обучающих событий")
    parser.add_argument("--neg-top", type=int, default=50, help="обучение: самых высоких негативов")
    parser.add_argument("--neg-random", type=int, default=50, help="обучение: случайных негативов")
    parser.add_argument("--max-train-events", type=int, default=0, help="0 — все")
    parser.add_argument("--max-val-events", type=int, default=0, help="0 — все (для проверок)")
    parser.add_argument("--out-dir", type=Path, default=RERANK, help=f"по умолчанию: {RERANK}")
    args = parser.parse_args(argv)
    args.nb_quota = args.k_nb / args.k
    args.quota = (args.k - args.k_geo - args.k_nb) / args.k

    t_start = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log(f"кандидаты для переранжировщика: части {', '.join(args.parts)}, K={args.k}: "
        f"гео {args.k_geo}, соседи до {args.nb_radius:g} км {args.k_nb}, "
        f"вне гео {args.k - args.k_geo - args.k_nb}")

    catalog_parts = [p for p in args.parts if p in ("train", "val")]
    if catalog_parts:
        from dense import CATALOG
        with step("читаем каталог и тексты"):
            items = load_catalog(TEXT_ITEM_COLUMNS + FILTER_ITEM_COLUMNS + GEO_COLUMNS)
            docs = doc_texts(items)
            items = items.drop(columns=TEXT_ITEM_COLUMNS)
        retrievers = build_retrievers(items, docs, CATALOG)
        del docs
        with step(f"соседние локации в радиусе {args.nb_radius:g} км"):
            args.neighbors = neighbor_map(corpus_centroids(items), args.nb_radius)
        print(f"  городов с соседями: {len(args.neighbors):,}")
        gc.collect()
        for part in catalog_parts:
            log(f"часть {part}")
            (run_val if part == "val" else run_train)(items, None, retrievers, args)
        del retrievers
        gc.collect()
    if "benchmark" in args.parts:
        log("часть benchmark")
        run_benchmark(args)

    log(f"готово за {(time.time() - t_start) / 60:.1f} мин")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

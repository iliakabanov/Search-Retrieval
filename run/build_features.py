"""Признаки пар «запрос — кандидат» для переранжировщика (src/features.py).

Читает кандидатов из dataset/rerank/ (run/build_candidates.py) и пишет
`<часть>_features.parquet`: qid, item_idx, label, служебные колонки
(fold / val_part, is_region) и признаки. Статистики по train — без утечки:
для train leave-one-event-out по train_pairs, для val — по train_pairs, для
benchmark — по train_pairs + val_pairs. Нужен torch_env: скоры dense на GPU.

    python run/build_features.py --parts train val
    python run/build_features.py --parts benchmark
"""

from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from bm25 import BM25
from data import (FILTER_ITEM_COLUMNS, TEXT_ITEM_COLUMNS, load_benchmark_items, load_catalog)
from features import (TrainStats, filter_features, haversine_km, item_features,
                      location_centroids, score_gaps)
from geo import neighbor_map
from logs import log, step
from paths import PROCESSED, RERANK, SPLIT
from texts import doc_texts

ITEM_COLUMNS = TEXT_ITEM_COLUMNS + FILTER_ITEM_COLUMNS + [
    "item_latitude", "item_longitude", "item_rating", "item_rating_reviews_count", "item_price",
    "item_microcat_id", "item_is_phone_hidden", "item_is_message_forbidden", "item_is_service"]
DENSE = ("e5-large", "RoSBERTa")


def stats_pairs(parts: list[str]) -> pd.DataFrame:
    """Пары train для статистик с микрокатегорией и локацией объявления."""
    pairs = pd.concat([pq.read_table(SPLIT / f"{p}.parquet", columns=[
        "event_id", "search_query", "search_location_id", "item_id"]).to_pandas() for p in parts])
    attrs = pq.read_table(PROCESSED / "train.parquet", columns=[
        "item_id", "item_microcat_id", "item_location_id"]).to_pandas().drop_duplicates("item_id")
    return pairs.merge(attrs, on="item_id", how="left")


def build_part(part: str, args) -> None:
    from dense import BENCHMARK, CATALOG, DenseRetriever

    events = pd.read_parquet(args.out_dir / f"{part}_events.parquet").set_index("qid")
    cands = pd.read_parquet(args.out_dir / f"{part}_candidates.parquet")
    print(f"  запросов: {len(events):,}, пар: {len(cands):,}")

    with step("читаем корпус и тексты"):
        if part == "benchmark":
            items = load_benchmark_items(ITEM_COLUMNS)
        else:
            items = load_catalog(ITEM_COLUMNS)
        docs = doc_texts(items)
        item_feats = item_features(items, items.item_title_raw.fillna("").str.len().to_numpy(),
                                   items.item_description_raw.fillna("").str.len().to_numpy())
        items = items.drop(columns=TEXT_ITEM_COLUMNS)

    texts = events["query"].unique()
    text_row = pd.Series(np.arange(len(texts)), index=texts)
    retrievers = {}
    with step("BM25: индекс по корпусу"):
        retrievers["bm25"] = BM25(docs)
    corpus = BENCHMARK if part == "benchmark" else CATALOG
    for name in DENSE:
        with step(f"{name}: эмбеддинги корпуса"):
            retrievers[name] = DenseRetriever(name, items.item_id, docs, corpus=corpus)
    del docs
    gc.collect()
    for name, r in retrievers.items():
        with step(f"{name}: готовим {len(texts):,} запросов"):
            r.prepare(texts)

    with step("статистики по train"):
        stats = TrainStats(stats_pairs(["train_pairs", "val_pairs"] if part == "benchmark"
                                       else ["train_pairs"]))
        train_items = pq.read_table(PROCESSED / "train.parquet", columns=[
            "item_location_id", "item_latitude", "item_longitude"]).to_pandas()
        centroids = location_centroids(
            pd.concat([train_items.item_location_id, items.item_location_id]),
            pd.concat([train_items.item_latitude, items.item_latitude]),
            pd.concat([train_items.item_longitude, items.item_longitude]))
        del train_items
        neighbors = neighbor_map(centroids, args.nb_radius)
        nb_pairs = pd.MultiIndex.from_tuples(
            [(q, i) for q, near in neighbors.items() for i in near])

    item_ids = items.item_id.to_numpy()
    item_loc = items.item_location_id.to_numpy()
    item_mc = items.item_microcat_id.to_numpy()
    item_lat, item_lon = items.item_latitude.to_numpy(), items.item_longitude.to_numpy()

    chunk_col = "fold" if part == "train" else None
    ev_extra = [c for c in ("fold", "val_part") if c in events]
    chunks = (cands.groupby(cands.qid.map(events[chunk_col]))
              if chunk_col else [(0, cands)])

    writer = None
    out_path = args.out_dir / f"{part}_features.parquet"
    for key, c in chunks:
        with step(f"признаки, порция {key}: {len(c):,} пар"):
            c = c.reset_index(drop=True)
            ev = events.loc[c.qid.to_numpy()]
            idx = c.item_idx.to_numpy()
            qrow = text_row.loc[ev["query"].to_numpy()].to_numpy()

            f = pd.DataFrame({"qid": c.qid, "item_idx": idx})
            if "label" in c:
                f["label"] = c.label.to_numpy()
            for col in ev_extra:
                f[col] = ev[col].to_numpy()
            f["is_region"] = ev.is_region.to_numpy().astype(np.int8)

            # текст
            f["sc_bm25"] = retrievers["bm25"].pair_scores(qrow, idx)
            for name in DENSE:
                f[f"sc_{name}"] = retrievers[name].pair_scores(qrow, idx)
            f = pd.concat([f, score_gaps(f, ["sc_bm25"] + [f"sc_{n}" for n in DENSE])], axis=1)
            f["sc_cand_rank"] = c.cand_rank.to_numpy()
            f["sc_source_quota"] = (c.source.to_numpy() == 1).astype(np.int8)
            f["sc_source_nb"] = (c.source.to_numpy() == 2).astype(np.int8)
            for col in [x for x in c.columns if x.startswith("rank_")]:
                f[f"sc_{col}"] = c[col].to_numpy()

            # гео
            sloc = ev.search_location_id.to_numpy()
            f["geo_same_loc"] = (item_loc[idx] == sloc).astype(np.int8)
            f["geo_is_neighbor"] = pd.MultiIndex.from_arrays([sloc, item_loc[idx]]).isin(
                nb_pairs).astype(np.int8)
            cen = centroids.reindex(sloc)
            f["geo_dist_km"] = haversine_km(cen.lat.to_numpy(), cen.lon.to_numpy(),
                                            item_lat[idx], item_lon[idx]).astype(np.float32)

            # объявление, фильтры, запрос
            f = pd.concat([f, item_feats.iloc[idx].reset_index(drop=True),
                           filter_features(c, events, items)], axis=1)
            q = ev["query"].to_numpy()
            f["q_words"] = np.array([len(t.split()) for t in q], dtype=np.int8)
            f["q_chars"] = np.array([len(t) for t in q], dtype=np.int16)

            # статистики по train
            st_in = pd.DataFrame({"qid": c.qid, "search_query": q, "search_location_id": sloc,
                                  "item_id": item_ids[idx], "item_microcat_id": item_mc[idx],
                                  "item_location_id": item_loc[idx]})
            f = pd.concat([f, stats.features(st_in, leave_one_out=(part == "train"))], axis=1)

        table = pa.Table.from_pandas(f, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, table.schema, compression="zstd")
        writer.write_table(table)
        del f, c
        gc.collect()
    writer.close()
    print(f"  записано: {out_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parts", nargs="+", default=["train", "val"],
                        choices=["train", "val", "benchmark"])
    parser.add_argument("--nb-radius", type=float, default=50.0,
                        help="радиус соседних локаций, км (как в build_candidates)")
    parser.add_argument("--out-dir", type=Path, default=RERANK,
                        help=f"где кандидаты и куда писать признаки (по умолчанию: {RERANK})")
    args = parser.parse_args(argv)
    t_start = time.time()
    log(f"признаки переранжировщика: части {', '.join(args.parts)}")
    for part in args.parts:
        log(f"часть {part}")
        build_part(part, args)
        gc.collect()
    log(f"готово за {(time.time() - t_start) / 60:.1f} мин")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

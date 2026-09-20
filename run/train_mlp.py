"""Обучение MLP-переранжировщика с lookup-эмбеддингами (src/mlp.py) и оценка.

Обучение — train_features, ранняя остановка по recall@50 на val-tune, итог —
val-test (docs/reranker_plan.md). Признаки — все из таблицы признаков, кроме
`--drop-features` (по умолчанию статистики конкретного объявления: на бенчмарке
90% объявлений не встречаются в train, и такие признаки туда не переносятся).
С `--compare-lgbm` в отчёт добавляются LightGBM из models/<имя> и ансамбль с ним.

Пишет в models/<имя>/: model.pt, prep.json (нормировка и словари),
per_event.parquet, recall.csv, config.json. Нужен torch_env и GPU.

    python run/train_mlp.py --text-emb e5-large --name mlp_nb_text
    python run/train_mlp.py --epochs 20 --hidden 512 256 --name mlp_big
    python run/train_mlp.py --text-emb e5-large --compare-lgbm lgbm_nb --name mlp_cmp

С --text-emb в сеть дополнительно подаются сами векторы запроса и объявления
(src/mlp.py, текстовая часть): векторы объявлений — из кэша эмбеддингов каталога,
запросов — кодируются и кэшируются в dataset/rerank/query_emb_*.
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
import pyarrow.parquet as pq
import torch
from data import load_catalog
from features import CATEGORICAL
from logs import log, step
from mlp import EMBEDDINGS, Preprocessor, RerankMLP, Vocab, listwise_loss
from paths import RERANK, RESULTS, ROOT
from rerank import (drop_groups_without_positives, feature_columns, format_report,
                    prepare_categorical, report, top_k_recall)

CAT_KEYS = list(EMBEDDINGS)


def load_part(part: str, data_dir: Path, numeric: list[str], items: pd.DataFrame) -> pd.DataFrame:
    """Признаки части + сырые категории для эмбеддингов."""
    path = data_dir / f"{part}_features.parquet"
    available = set(pq.ParquetFile(path).schema_arrow.names)
    cols = ["qid", "item_idx", "it_microcat"] + numeric + [
        c for c in ("label", "val_part") if c in available]     # у бенчмарка меток нет
    df = pd.read_parquet(path, columns=cols)
    events = pd.read_parquet(data_dir / f"{part}_events.parquet",
                             columns=["qid", "query", "search_location_id", "filter_category"]
                             ).set_index("qid")
    idx = df.item_idx.to_numpy()
    ev = events.loc[df.qid.to_numpy()]
    df["microcat"] = df.pop("it_microcat")
    df["category"] = items.item_category.fillna("").to_numpy()[idx]
    df["item_loc"] = items.item_location_id.to_numpy()[idx]
    df["query_loc"] = ev.search_location_id.to_numpy()
    df["query_category"] = ev.filter_category.fillna("").to_numpy()
    df["query_text"] = ev["query"].to_numpy()
    return df


class TextVectors:
    """Векторы запросов и объявлений на GPU для текстовой части MLP.

    `rows(df)` — номера строк (запрос, объявление) для каждой пары df.
    """

    def __init__(self, name: str, texts, item_ids, corpus: str, cache_tag: str):
        import dense

        self.texts = pd.Index(pd.unique(np.asarray(texts, dtype=object)))
        self.q = torch.from_numpy(dense.query_embeddings(
            name, self.texts.to_numpy(), RERANK / f"query_emb_{name}_{cache_tag}")).cuda()
        self.d = torch.from_numpy(dense.catalog_embeddings(name, item_ids, corpus=corpus)).cuda()

    def rows(self, df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        return (torch.from_numpy(self.texts.get_indexer(df.query_text.to_numpy()).astype(np.int64)),
                torch.from_numpy(df.item_idx.to_numpy(dtype=np.int64)))


def tensors(df: pd.DataFrame, prep: Preprocessor, vocabs: dict[str, Vocab]):
    num = torch.from_numpy(prep.transform(df))
    cats = {k: torch.from_numpy(vocabs[k].encode(df[k].to_numpy())) for k in CAT_KEYS}
    return num, cats


@torch.no_grad()
def predict(model, num, cats, text=None, batch: int = 200_000) -> np.ndarray:
    """`text` — (TextVectors, номера запросов, номера объявлений) для текстовой части."""
    model.eval()
    out = []
    for s in range(0, len(num), batch):
        vecs = {}
        if text is not None:
            tv, q_rows, d_rows = text
            vecs = {"q_vec": tv.q[q_rows[s:s + batch].cuda()],
                    "d_vec": tv.d[d_rows[s:s + batch].cuda()]}
        out.append(model(num[s:s + batch].cuda().float(),
                         {k: v[s:s + batch].cuda().long() for k, v in cats.items()},
                         **vecs).float().cpu())
    return torch.cat(out).numpy()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=RERANK)
    parser.add_argument("--drop-features", nargs="*",
                        default=["st_item_pop_log", "st_text_item"],
                        help="признаки, которые не давать модели; по умолчанию статистики "
                             "конкретного объявления — на бенчмарке 90%% объявлений новые")
    parser.add_argument("--compare-lgbm", default=None,
                        help="модель LightGBM в models/ для сравнения и ансамбля (необязательно)")
    parser.add_argument("--baseline", default="retrievers_regions")
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 128])
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--groups-per-batch", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--text-emb", default=None, choices=[None, "e5-large", "RoSBERTa"],
                        help="подавать в сеть векторы запроса и объявления этой модели")
    parser.add_argument("--text-dim", type=int, default=64,
                        help="до скольких чисел сжимать векторы")
    parser.add_argument("--name", default="mlp")
    args = parser.parse_args(argv)
    t_start = time.time()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    log(f"MLP-переранжировщик: {args.name}")

    numeric = feature_columns(pq.ParquetFile(args.data_dir / "train_features.parquet")
                              .schema_arrow.names)
    numeric = [f for f in numeric if f not in CATEGORICAL and f not in set(args.drop_features)]

    with step("читаем признаки и категории"):
        items = load_catalog(["item_category", "item_location_id"])
        train = drop_groups_without_positives(load_part("train", args.data_dir, numeric, items))
        train = train.sort_values("qid", kind="stable").reset_index(drop=True)
        val = load_part("val", args.data_dir, numeric, items)
        val_events = pd.read_parquet(args.data_dir / "val_events.parquet").set_index("qid")
    print(f"  обучение: {train.qid.nunique():,} событий, {len(train):,} пар; "
          f"числовых признаков: {len(numeric)}; эмбеддинги: {', '.join(CAT_KEYS)}")

    with step("нормировка и словари категорий (по обучению)"):
        prep = Preprocessor(numeric).fit(train)
        vocabs = {k: Vocab(train[k].to_numpy()) for k in CAT_KEYS}
        num_tr, cats_tr = tensors(train, prep, vocabs)
        num_va, cats_va = tensors(val, prep, vocabs)
        text_tr = text_va = None
        if args.text_emb:
            from dense import CATALOG
            tv = TextVectors(args.text_emb, np.concatenate([train.query_text.unique(),
                                                             val.query_text.unique()]),
                             items.item_id, CATALOG, "train_val")
            text_tr, text_va = (tv, *tv.rows(train)), (tv, *tv.rows(val))
            print(f"  векторы {args.text_emb}: запросов {len(tv.texts):,}, объявлений {len(tv.d):,}")
        labels_tr = torch.from_numpy(train.label.to_numpy(dtype=np.float32))
        # группы: индексы строк каждого события, дополненные до одной длины
        starts = np.flatnonzero(np.r_[True, train.qid.to_numpy()[1:] != train.qid.to_numpy()[:-1]])
        sizes = np.diff(np.r_[starts, len(train)])
        max_size = int(sizes.max())
        pad = np.full((len(starts), max_size), -1, dtype=np.int64)
        for i, (s, n) in enumerate(zip(starts, sizes)):
            pad[i, :n] = np.arange(s, s + n)
        del train
        gc.collect()
    print(f"  вход сети: {prep.n_out} числовых + эмбеддинги; групп {len(pad):,}, до {max_size} кандидатов")
    print("  размеры словарей: " + ", ".join(f"{k} {len(v):,}" for k, v in vocabs.items()))

    model = RerankMLP(prep.n_out, {k: len(v) for k, v in vocabs.items()}, args.hidden,
                      args.dropout, text_dim=args.text_dim if args.text_emb else 0).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    tune_mask = (val.val_part == "tune").to_numpy()
    tune_df = val[tune_mask]
    best, best_state, bad_epochs = -1.0, None, 0

    for epoch in range(1, args.epochs + 1):
        with step(f"эпоха {epoch}/{args.epochs}"):
            model.train()
            order = rng.permutation(len(pad))
            losses = []
            for b in range(0, len(order), args.groups_per_batch):
                rows = torch.from_numpy(pad[order[b:b + args.groups_per_batch]])
                mask = rows >= 0
                flat = rows.clamp(min=0).reshape(-1)
                vecs = {}
                if text_tr is not None:
                    tv, q_rows, d_rows = text_tr
                    vecs = {"q_vec": tv.q[q_rows[flat].cuda()], "d_vec": tv.d[d_rows[flat].cuda()]}
                scores = model(num_tr[flat].cuda().float(),
                               {k: v[flat].cuda().long() for k, v in cats_tr.items()}, **vecs)
                loss = listwise_loss(scores.view(rows.shape), labels_tr[flat].cuda().view(rows.shape),
                                     mask.cuda())
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(loss.item())
            tune_text = None if text_va is None else (
                text_va[0], text_va[1][tune_mask], text_va[2][tune_mask])
            tune_score = predict(model, num_va[tune_mask],
                                 {k: v[tune_mask] for k, v in cats_va.items()}, tune_text)
            tune_recall = top_k_recall(tune_df, tune_score, val_events.n_pos[
                val_events.val_part == "tune"], args.top_k).mean()
        print(f"  loss {np.mean(losses):.4f} | recall@{args.top_k} val-tune {tune_recall:.2%}")
        if tune_recall > best:
            best, bad_epochs = tune_recall, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                log(f"ранняя остановка: {args.patience} эпохи без улучшения")
                break
    model.load_state_dict(best_state)

    with step("оценка на валидации"):
        scores = {"MLP": predict(model, num_va, cats_va, text_va)}
        if args.compare_lgbm:      # необязательное сравнение с бустингом и ансамбль с ним
            import lightgbm as lgb

            lgb_dir = ROOT / "models" / args.compare_lgbm
            lgb_config = json.loads((lgb_dir / "config.json").read_text(encoding="utf-8"))
            booster = lgb.Booster(model_str=(lgb_dir / "model.txt").read_text(encoding="utf-8"))
            scores["LightGBM"] = booster.predict(prepare_categorical(
                pd.read_parquet(args.data_dir / "val_features.parquet",
                                columns=lgb_config["features"])))
            rank = lambda s: pd.Series(s).groupby(val.qid.to_numpy()).rank(pct=True).to_numpy()
            scores["ансамбль"] = rank(scores["MLP"]) + rank(scores["LightGBM"])
        base = pd.read_parquet(RESULTS / args.baseline / "per_event.parquet")[
            "RRF | гео + добивка"].reindex(val_events.index)
        table, per_event = report(val, scores, val_events, base, args.top_k)
    print(f"\nrecall@{args.top_k}")
    print(format_report(table))

    out = ROOT / "models" / args.name
    with step(f"сохраняем модель и отчёт в {out}"):
        out.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out / "model.pt")
        (out / "prep.json").write_text(json.dumps(
            {"prep": prep.to_json(), "vocabs": {k: v.to_json() for k, v in vocabs.items()}},
            ensure_ascii=False), encoding="utf-8")
        per_event.to_parquet(out / "per_event.parquet")
        table.to_csv(out / "recall.csv", encoding="utf-8-sig")
        (out / "config.json").write_text(json.dumps(
            {"type": "mlp", **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
             "numeric": numeric, "best_tune_recall": best}, ensure_ascii=False, indent=2),
            encoding="utf-8")
    log(f"готово за {(time.time() - t_start) / 60:.1f} мин")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

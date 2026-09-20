"""Ответ на бенчмарк переранжировщиком: топ-50 из кандидатов по скору модели.

Нужны кандидаты и признаки бенчмарка (run/build_candidates.py --parts benchmark,
run/build_features.py --parts benchmark) и обученная модель: MLP (run/train_mlp.py)
и/или LightGBM (run/train_reranker.py). Модель берёт те же признаки, что при
обучении (список — в models/<имя>/config.json).

Пишет в results/<имя>/: answer.csv (проверяется на требования задания),
predictions.parquet (query_id, rank, item_id, score), config.json. Печатает,
насколько ответ совпадает с прошлым (RRF), — для контроля.

    python run/rerank_benchmark.py --mlp mlp_nb_text                 # итоговое решение
    python run/rerank_benchmark.py --model lgbm_nb                   # только LightGBM
    python run/rerank_benchmark.py --mlp mlp_nb_text --model lgbm_nb # ансамбль

Если заданы обе модели, скор — сумма их рангов (в долях) внутри запроса, как в
ансамбле на валидации (run/train_mlp.py).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pandas as pd
from answer import check_answer, make_answer, save_answer
from data import load_benchmark_items, load_benchmark_queries
from logs import log, step
from paths import RERANK, RESULTS, ROOT
from rerank import prepare_categorical


def mlp_scores(name: str, data_dir: Path) -> np.ndarray:
    """Скоры MLP для строк benchmark_features (в их порядке)."""
    import torch
    from mlp import Preprocessor, RerankMLP, Vocab
    from train_mlp import TextVectors, load_part, predict, tensors

    folder = ROOT / "models" / name
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    saved = json.loads((folder / "prep.json").read_text(encoding="utf-8"))
    prep = Preprocessor.from_json(saved["prep"])
    vocabs = {k: Vocab(np.array(v, dtype=object) if v and isinstance(v[0], str) else np.array(v))
              for k, v in saved["vocabs"].items()}
    items = load_benchmark_items(["item_category", "item_location_id"])
    df = load_part("benchmark", data_dir, config["numeric"], items)
    text_emb = config.get("text_emb")
    model = RerankMLP(prep.n_out, {k: len(v) for k, v in vocabs.items()}, config["hidden"],
                      config["dropout"], text_dim=config["text_dim"] if text_emb else 0).cuda()
    model.load_state_dict(torch.load(folder / "model.pt"))
    num, cats = tensors(df, prep, vocabs)
    text = None
    if text_emb:
        from dense import BENCHMARK
        tv = TextVectors(text_emb, df.query_text.unique(), items.item_id, BENCHMARK,
                         "benchmark", data_dir)
        text = (tv, *tv.rows(df))
    return predict(model, num, cats, text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mlp", default=None, help="папка в models/ с MLP")
    parser.add_argument("--model", default=None,
                        help="папка в models/ с LightGBM (без --mlp — ответ только по нему, "
                             "вместе с --mlp — ансамбль)")
    parser.add_argument("--data-dir", type=Path, default=RERANK)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--compare", default="benchmark",
                        help="папка results/ с прошлым ответом для сравнения")
    parser.add_argument("--name", default=None, help="папка результатов (по умолчанию benchmark_<модель>)")
    args = parser.parse_args(argv)
    if not args.mlp and not args.model:
        parser.error("нужен --mlp и/или --model")
    name = args.name or "benchmark_" + "_".join(m for m in (args.model, args.mlp) if m)
    t_start = time.time()
    log(f"ответ на бенчмарк: {', '.join(m for m in (args.mlp, args.model) if m)}, "
        f"топ-{args.top_k}")

    with step("читаем признаки и корпус"):
        feats = pd.read_parquet(args.data_dir / "benchmark_features.parquet")
        queries = load_benchmark_queries()
        item_ids = load_benchmark_items([]).item_id.to_numpy()
    print(f"  запросов: {feats.qid.nunique():,} из {len(queries):,}, пар: {len(feats):,}")

    with step("скоры и топ по запросу"):
        rank = lambda s: pd.Series(s).groupby(feats.qid.to_numpy()).rank(pct=True).to_numpy()
        parts = {}
        if args.mlp:
            parts["mlp"] = mlp_scores(args.mlp, args.data_dir)
        features = None
        if args.model:
            import lightgbm as lgb      # нужен только для варианта с бустингом

            model_dir = ROOT / "models" / args.model
            config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
            features = config["features"]
            booster = lgb.Booster(model_str=(model_dir / "model.txt").read_text(encoding="utf-8"))
            parts["lgbm"] = booster.predict(prepare_categorical(feats[features].copy()))
        # одна модель — её скор; две — сумма рангов внутри запроса, как на валидации
        feats["score"] = (list(parts.values())[0] if len(parts) == 1
                          else sum(rank(s) for s in parts.values()))
        feats = feats.sort_values(["qid", "score"], ascending=[True, False], kind="stable")
        feats["rank"] = feats.groupby("qid").cumcount() + 1
        top = feats[feats["rank"] <= args.top_k]
        predictions = pd.DataFrame({"query_id": top.qid.to_numpy(), "rank": top["rank"].to_numpy(),
                                    "item_id": item_ids[top.item_idx.to_numpy()],
                                    "score": top.score.to_numpy()})

    with step("проверяем answer.csv на требования задания"):
        answer = make_answer(predictions.groupby("query_id", sort=False).item_id.apply(list)
                             .to_dict(), queries.query_id)
        problems = check_answer(answer, queries.query_id, item_ids)
    for p in problems:
        log(f"ОШИБКА в ответе: {p}")
    if problems:
        return 1

    n = predictions.groupby("query_id").size().reindex(queries.query_id, fill_value=0)
    print(f"  полный топ-{args.top_k} у {(n == args.top_k).sum():,} запросов, "
          f"короче у {((n > 0) & (n < args.top_k)).sum():,}, пусто у {(n == 0).sum():,}")
    prev = RESULTS / args.compare / "answer.csv"
    if prev.exists():
        old = pd.read_csv(prev, dtype=str).set_index("query_id").answer.fillna("").str.split()
        new = answer.set_index("query_id").answer.str.split()
        overlap = pd.Series({q: len(set(old[q]) & set(new[q])) / max(len(new[q]), 1)
                             for q in new.index})
        print(f"  совпадение с прошлым ответом ({args.compare}): {overlap.mean():.1%} в среднем")

    out = RESULTS / name
    with step(f"сохраняем в {out}"):
        out.mkdir(parents=True, exist_ok=True)
        save_answer(answer, out / "answer.csv")
        predictions.to_parquet(out / "predictions.parquet", index=False)
        (out / "config.json").write_text(json.dumps(
            {"model": args.model, "mlp": args.mlp, "top_k": args.top_k, "features": features},
            ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"answer.csv: {out / 'answer.csv'}")
    log(f"готово за {(time.time() - t_start) / 60:.1f} мин")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

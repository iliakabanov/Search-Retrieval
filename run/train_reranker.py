"""Обучение переранжировщика LightGBM (LambdaRank) и оценка на валидации.

Обучение — train_features (события train_pairs), ранняя остановка — val-tune,
итог — val-test (docs/reranker_plan.md). Сравнение с нынешним методом (RRF,
«гео + добивка») на тех же событиях — по results/<--baseline>/per_event.parquet.

Пишет модель и отчёт в models/<имя>/: model.txt, feature_importance.csv,
per_event.parquet (recall переранжировщика и базы по событиям валидации), config.json.

    python run/train_reranker.py
    python run/train_reranker.py --max-train-events 60000 --name lgbm_60k
    python run/train_reranker.py --drop-features st_item_pop_log --name lgbm_no_pop
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import lightgbm as lgb
import numpy as np
import pandas as pd
from features import CATEGORICAL
from logs import log, step
from paths import RERANK, RESULTS, ROOT
from rerank import (drop_groups_without_positives, feature_columns, group_sizes,
                    prepare_categorical, top_k_recall)


def pct(x: float) -> str:
    return f"{x:.2%}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=RERANK)
    parser.add_argument("--baseline", default="retrievers_regions",
                        help="папка results/ с per_event.parquet нынешнего метода")
    parser.add_argument("--max-train-events", type=int, default=0, help="0 — все")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--min-data-in-leaf", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=3000)
    parser.add_argument("--early-stopping", type=int, default=100)
    parser.add_argument("--drop-features", nargs="*", default=[],
                        help="признаки, которые не давать модели (например st_item_pop_log)")
    parser.add_argument("--name", default="lgbm")
    args = parser.parse_args(argv)
    t_start = time.time()
    log(f"переранжировщик LightGBM LambdaRank: {args.name}")

    with step("читаем признаки"):
        train = pd.read_parquet(args.data_dir / "train_features.parquet")
        val = pd.read_parquet(args.data_dir / "val_features.parquet")
        val_events = pd.read_parquet(args.data_dir / "val_events.parquet").set_index("qid")
    if args.max_train_events:
        keep = np.random.default_rng(0).choice(train.qid.unique(), args.max_train_events,
                                               replace=False)
        train = train[train.qid.isin(set(keep))]
    n_train_events = train.qid.nunique()
    train = drop_groups_without_positives(train).sort_values(["qid", "item_idx"], kind="stable")
    features = [f for f in feature_columns(train) if f not in set(args.drop_features)]
    unknown = set(args.drop_features) - set(feature_columns(train))
    if unknown:
        parser.error(f"нет таких признаков: {', '.join(sorted(unknown))}")
    print(f"  обучение: {n_train_events:,} событий -> {train.qid.nunique():,} с позитивом "
          f"среди кандидатов, {len(train):,} пар; признаков: {len(features)}"
          + (f" (без {', '.join(args.drop_features)})" if args.drop_features else ""))

    tune = val[val.val_part == "tune"].sort_values(["qid", "item_idx"], kind="stable")
    test = val[val.val_part == "test"]
    tune_fit = drop_groups_without_positives(tune)
    print(f"  val-tune: {tune.qid.nunique():,} событий, val-test: {test.qid.nunique():,}")

    params = {
        "objective": "lambdarank", "metric": "ndcg", "eval_at": [args.top_k],
        "lambdarank_truncation_level": args.top_k + 10,
        "learning_rate": args.learning_rate, "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_data_in_leaf, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
        "verbosity": -1, "num_threads": 0, "seed": 0,
    }
    with step("готовим датасеты LightGBM"):
        dtrain = lgb.Dataset(prepare_categorical(train[features].copy()), train.label,
                             group=group_sizes(train),
                             categorical_feature=[c for c in CATEGORICAL if c in features],
                             free_raw_data=True)
        dtune = lgb.Dataset(prepare_categorical(tune_fit[features].copy()), tune_fit.label,
                            group=group_sizes(tune_fit), reference=dtrain)
        del train
        gc.collect()

    with step(f"обучение (до {args.rounds} деревьев, ранняя остановка {args.early_stopping})"):
        model = lgb.train(params, dtrain, args.rounds, valid_sets=[dtune], valid_names=["tune"],
                          callbacks=[lgb.early_stopping(args.early_stopping, verbose=False),
                                     lgb.log_evaluation(100)])
    print(f"  лучшая итерация: {model.best_iteration}")

    with step("оценка на валидации"):
        score = model.predict(prepare_categorical(val[features].copy()),
                              num_iteration=model.best_iteration)
        recall = top_k_recall(val, score, val_events.n_pos, args.top_k)
        cand_recall = top_k_recall(val, np.zeros(len(val)), val_events.n_pos, 10 ** 6)
        base = pd.read_parquet(RESULTS / args.baseline / "per_event.parquet")
        base = base["RRF | гео + добивка"].reindex(val_events.index)
        rep = val_events[["val_part", "is_region", "slice"]].assign(
            reranker=recall, baseline=base, candidates=cand_recall)

    rows = {}
    for part in ("tune", "test"):
        p = rep[rep.val_part == part]
        rows[f"val-{part}"] = {"RRF (сейчас)": p.baseline.mean(), "переранжировщик": p.reranker.mean(),
                               "потолок кандидатов": p.candidates.mean()}
        for grp, mask in (("город", ~p.is_region), ("регион", p.is_region)):
            q = p[mask]
            rows[f"val-{part}, {grp}"] = {"RRF (сейчас)": q.baseline.mean(),
                                          "переранжировщик": q.reranker.mean(),
                                          "потолок кандидатов": q.candidates.mean()}
    table = pd.DataFrame(rows).T
    table["прирост, п.п."] = (table["переранжировщик"] - table["RRF (сейчас)"]) * 100
    print(f"\nrecall@{args.top_k}")
    print(table.to_string(formatters={"RRF (сейчас)": pct, "переранжировщик": pct,
                                      "потолок кандидатов": pct,
                                      "прирост, п.п.": "{:+.2f}".format}))

    importance = pd.DataFrame({"feature": model.feature_name(),
                               "gain": model.feature_importance("gain")}).sort_values(
        "gain", ascending=False)
    importance["gain_share"] = importance.gain / importance.gain.sum()
    print("\nважность признаков (топ-15 по gain)")
    print(importance.head(15).to_string(index=False, formatters={"gain": "{:,.0f}".format,
                                                                 "gain_share": pct}))

    out = ROOT / "models" / args.name
    with step(f"сохраняем модель и отчёт в {out}"):
        out.mkdir(parents=True, exist_ok=True)
        # LightGBM не пишет по пути с кириллицей — сохраняем строку модели сами
        (out / "model.txt").write_text(model.model_to_string(num_iteration=model.best_iteration),
                                       encoding="utf-8")
        importance.to_csv(out / "feature_importance.csv", index=False)
        rep.to_parquet(out / "per_event.parquet")
        table.to_csv(out / "recall.csv", encoding="utf-8-sig")
        (out / "config.json").write_text(json.dumps(
            {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
             "best_iteration": model.best_iteration, "features": features, "params": params},
            ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"готово за {(time.time() - t_start) / 60:.1f} мин")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

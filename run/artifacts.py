"""Скачивает готовые артефакты с Hugging Face, чтобы не ждать пересчёта.

Артефакты **не нужны** для работы решения: без них всё считается локально теми же
командами (README), просто дольше. Скрипт знает, какие файлы репозитория куда
положить, чтобы остальной пайплайн нашёл их на привычных местах.

Наборы:

* `rerank` (~51 МБ) — кандидаты, признаки и векторы запросов бенчмарка в
  `dataset/rerank/`. С ними ответ собирается одной командой
  `run/rerank_benchmark.py --mlp mlp_nb_text` за минуту.
* `train` (~1.0 ГБ) — признаки и события train и val: с ними можно обучить
  переранжировщик заново (`run/train_mlp.py`) и повторить метрики на валидации.
* `embeddings` (~1.5 ГБ) — эмбеддинги каталога валидации и корпуса бенчмарка
  обеими dense-моделями в `dataset/embeddings/`: экономит ~4 часа кодирования на GPU.
* `e5-catalog` и `e5-benchmark` (по ~382 МБ) — части того же кэша: только векторы
  e5-large для каталога валидации и для корпуса бенчмарка. Столько нужно, если
  кандидаты не пересчитываются (RoSBERTa нужна только им), а векторы e5 идут в
  текстовую часть MLP: `e5-catalog` — для `run/train_mlp.py --text-emb e5-large`,
  `e5-benchmark` — для `run/rerank_benchmark.py --mlp mlp_nb_text`.

    python run/artifacts.py --sets rerank
    python run/artifacts.py --sets rerank train
"""

from __future__ import annotations

import argparse
import sys
from fnmatch import fnmatch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from logs import log, step
from paths import EMBEDDINGS, RERANK

#: Набор -> (локальная папка, шаблоны файлов). В репозитории файлы лежат в корне
#: с теми же именами, поэтому скачанное просто раскладывается по локальным папкам.
SETS = {
    "rerank": (RERANK, ["benchmark_candidates.parquet", "benchmark_events.parquet",
                        "benchmark_features.parquet", "query_emb_e5-large_benchmark.npy",
                        "query_emb_e5-large_benchmark.texts.npy"]),
    # признаки и события train и val — всё, что нужно, чтобы обучить переранжировщик
    # заново и повторить метрики на валидации (кандидаты для этого не нужны)
    "train": (RERANK, ["train_features.parquet", "val_features.parquet",
                       "train_events.parquet", "val_events.parquet",
                       "query_emb_e5-large_train_val.npy",
                       "query_emb_e5-large_train_val.texts.npy"]),
    # кэш эмбеддингов каталога валидации и корпуса бенчмарка
    "embeddings": (EMBEDDINGS, ["e5-large/*", "RoSBERTa/*", "benchmark/*"]),
    # только e5: векторы текстовой части MLP, без RoSBERTa (она нужна лишь кандидатам)
    "e5-catalog": (EMBEDDINGS, ["e5-large/*"]),
    "e5-benchmark": (EMBEDDINGS, ["benchmark/e5-large/*"]),
}


def folder_size(path: Path, patterns: list[str]) -> str:
    """Сколько весит скачанное (шаблоны — как их понимает huggingface_hub)."""
    files = [f for f in path.rglob("*") if f.is_file()
             and any(fnmatch(f.relative_to(path).as_posix(), p) for p in patterns)]
    return f"{sum(f.stat().st_size for f in files) / 1024 ** 2:,.0f} МБ"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-id", default="iliakabanov/search-retrieval",
                        help="репозиторий-датасет на Hugging Face")
    parser.add_argument("--sets", nargs="+", default=["rerank"], choices=list(SETS))
    parser.add_argument("--retries", type=int, default=5, help="попыток при обрыве сети")
    args = parser.parse_args(argv)

    import time

    from huggingface_hub import snapshot_download

    log(f"скачиваем с {args.repo_id}: {', '.join(args.sets)}")
    for name in args.sets:
        local, patterns = SETS[name]
        local.mkdir(parents=True, exist_ok=True)
        with step(f"{name} -> {local}"):
            for attempt in range(1, args.retries + 1):
                try:
                    snapshot_download(repo_id=args.repo_id, repo_type="dataset",
                                      allow_patterns=patterns, local_dir=str(local))
                    break
                except Exception as e:      # обрыв сети — уже скачанное не теряется
                    log(f"попытка {attempt}/{args.retries}: {type(e).__name__}")
                    if attempt == args.retries:
                        raise
                    time.sleep(5)
        print(f"  {name}: {folder_size(local, patterns)}")
    log("готово")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

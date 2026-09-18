"""Разбиение train на обучение и валидацию и каталог для поиска (см. src/split.py).

Пишет `dataset/split/`. После смены разбиения кэш эмбеддингов каталога
пересчитается сам при следующем запуске (он сверяет список item_id).

    python run/make_split.py
    python run/make_split.py --seed 7 --catalog-mode keep_positives
    python run/make_split.py --out dataset/split_test
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import numpy as np
import pandas as pd
from logs import log, step
from paths import SPLIT
from split import (CATALOG_MODES, SEEN, Split, SplitConfig, load_pairs, report,
                   sample_catalog, save_split, select_eval, split_by_text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    defaults = SplitConfig()
    for f in fields(SplitConfig):
        flag = "--" + f.name.replace("_", "-")
        default = getattr(defaults, f.name)
        if isinstance(default, bool):
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=default)
        else:
            parser.add_argument(flag, type=type(default), default=default,
                                choices=CATALOG_MODES if f.name == "catalog_mode" else None,
                                help=f"по умолчанию: {default}")
    parser.add_argument("--out", type=Path, default=SPLIT, help=f"по умолчанию: {SPLIT}")
    args = parser.parse_args(argv)
    cfg = SplitConfig(**{f.name: getattr(args, f.name) for f in fields(SplitConfig)})
    pd.set_option("display.width", 200)

    log(f"разбиение: seed={cfg.seed}, каталог {cfg.catalog_mode} на {cfg.catalog_size:,}")
    # тот же порядок вызовов генератора, что в split.make_split, — иначе разбиение другое
    rng = np.random.default_rng(cfg.seed)

    with step("читаем пары «событие → объявление» из train и схлопываем повторы"):
        pairs = load_pairs(cfg.dedup)
    print(f"  примеров: {len(pairs):,} | событий: {pairs.event.nunique():,} | "
          f"текстов запроса: {pairs.search_query.nunique():,}")

    with step("делим по тексту запроса на train и val"):
        train_pairs, val_pairs = split_by_text(pairs, cfg, rng)
    share = val_pairs[val_pairs.slice == SEEN].event.nunique() / val_pairs.event.nunique()
    print(f"  доля val-событий со знакомым текстом: {share:.1%} (цель {cfg.seen_text_share:.0%})")

    with step("собираем каталог"):
        catalog = sample_catalog(pairs, val_pairs, cfg, rng)
    val_items = set(val_pairs.item_id)
    in_catalog = len(val_items & catalog)
    print(f"  каталог: {pairs.item_id.nunique():,} -> {len(catalog):,} | позитивов val в "
          f"каталоге: {in_catalog:,} из {len(val_items):,} ({in_catalog / len(val_items):.1%})")

    with step("отбираем события валидации для оценки"):
        val_eval = select_eval(val_pairs, catalog, cfg)
    split = Split(train_pairs, val_pairs, val_eval, catalog, pairs.item_id.nunique())

    with step(f"сохраняем в {args.out}"):
        save_split(split, cfg, args.out)

    print()
    print(report(split, cfg).to_string(formatters={c: "{:,}".format for c in
                                                   ("событий", "примеров", "текстов")}))
    log("готово")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

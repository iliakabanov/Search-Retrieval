"""Предобработка сырых parquet из `dataset/` в `dataset/processed/` (см. src/preprocess.py).

    python run/preprocess.py
    python run/preprocess.py --data-dir dataset --out-dir dataset/processed
    python run/preprocess.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import preprocess as pp
import pyarrow.parquet as pq
from logs import log, step
from paths import DATASET


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DATASET,
                        help="каталог с сырыми parquet-файлами (по умолчанию: dataset)")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="куда писать результат (по умолчанию: <data-dir>/processed)")
    parser.add_argument("--batch-size", type=int, default=pp.BATCH_SIZE,
                        help=f"размер батча при потоковой обработке (по умолчанию: {pp.BATCH_SIZE})")
    parser.add_argument("--dry-run", action="store_true",
                        help="показать план изменений, ничего не записывая")
    args = parser.parse_args(argv)

    data_dir: Path = args.data_dir
    out_dir: Path = args.out_dir or data_dir / "processed"

    missing = [n for n in pp.DATASETS if not (data_dir / f"{n}.parquet").exists()]
    if missing:
        parser.error(f"в {data_dir} не найдены файлы: {', '.join(missing)}. "
                     "Укажите верный --data-dir.")

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    log("предобработка датасета")
    print(f"вход:  {data_dir}")
    print(f"выход: {out_dir}{' (dry-run, запись отключена)' if args.dry_run else ''}\n")

    with step("предобработка всех файлов"):
        for i, name in enumerate(pp.DATASETS, start=1):
            src = data_dir / f"{name}.parquet"
            dst = out_dir / f"{name}.parquet"
            n_rows = pq.ParquetFile(src).metadata.num_rows
            changes = pp.describe_changes(src)

            print(f"\n{name} ({i}/{len(pp.DATASETS)}), строк: {n_rows:,}")
            print(f"  удалено колонок: {', '.join(changes['dropped']) or '—'}")
            print(f"  приведено к float64: {', '.join(changes['casted']) or '—'}")
            print(f"  добавлено колонок: {', '.join(changes['added']) or '—'}")

            if args.dry_run:
                continue

            with step(f"{name}: разбираем infm_params и пишем {dst.name}"):
                rows = pp.process_file(src, dst, batch_size=args.batch_size)
            size_mb = dst.stat().st_size / 1024 ** 2
            print(f"  записано: {dst} ({rows:,} строк, {size_mb:,.1f} МБ)")

    log("готово")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

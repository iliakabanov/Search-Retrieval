"""Предобработка датасета Search-Retrieval.

Приводит сырые parquet-файлы из `dataset/` к рабочему виду и складывает
результат в `dataset/processed/`.

Что делает на текущем шаге:

1. Удаляет вырожденные колонки запроса (`search_category`,
   `search_is_delivery_search`) — см. обоснование в `notebooks/01_eda.ipynb`.
   Дроп применяется ко всем файлам, где эти колонки есть, включая `train`,
   чтобы набор признаков на обучении и на инференсе совпадал.
2. Приводит `decimal128`-колонки (`item_price`, `item_latitude`,
   `item_longitude`) к `float64` — в pandas они иначе приходят объектами
   `Decimal`, с которыми не работает ни арифметика, ни модели.

Файлы обрабатываются потоково (batch → batch), поэтому пиковая память не
зависит от размера входа.

Запуск:
    python src/preprocess.py
    python src/preprocess.py --data-dir dataset --out-dir dataset/processed
    python src/preprocess.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

# --- конфигурация ------------------------------------------------------------

#: Вырожденные колонки запроса. Обоснование (EDA, раздел 3.2 и 9):
#:  * search_is_delivery_search — 0 у 100% бенчмарка, 1 лишь у 12 строк train;
#:  * search_category — в бенчмарке 9.1% значений `0`, в train 0.007%, причём
#:    `0` не означает другой домен (позитивы всё равно из категории 114).
#:    Оставлять её опаснее, чем убрать: модель выучит константу 114 и получит
#:    на инференсе 9% практически невиданного значения.
DROP_COLUMNS: tuple[str, ...] = (
    "search_category",
    "search_is_delivery_search",
)

#: decimal128 → float64.
DECIMAL_COLUMNS: tuple[str, ...] = (
    "item_price",
    "item_latitude",
    "item_longitude",
)

#: Имена файлов датасета (без расширения).
DATASETS: tuple[str, ...] = (
    "benchmark_queries",
    "benchmark_items",
    "train",
)

BATCH_SIZE = 50_000
COMPRESSION = "zstd"


# --- преобразования ----------------------------------------------------------


def drop_degenerate(table: pa.Table) -> pa.Table:
    """Удаляет вырожденные колонки, если они присутствуют."""
    present = [c for c in DROP_COLUMNS if c in table.column_names]
    return table.drop_columns(present) if present else table


def cast_decimals(table: pa.Table) -> pa.Table:
    """Приводит decimal128-колонки к float64."""
    for name in DECIMAL_COLUMNS:
        if name not in table.column_names:
            continue
        idx = table.column_names.index(name)
        column = table.column(name)
        if pa.types.is_decimal(column.type):
            table = table.set_column(
                idx, name, pc.cast(column, pa.float64())
            )
    return table


def transform(table: pa.Table) -> pa.Table:
    """Полный конвейер преобразований одного батча."""
    return cast_decimals(drop_degenerate(table))


# --- ввод/вывод --------------------------------------------------------------


def describe_changes(src: Path) -> dict[str, list[str]]:
    """Что именно изменится в файле — считается по схеме, без чтения данных."""
    schema = pq.ParquetFile(src).schema_arrow
    return {
        "dropped": [c for c in DROP_COLUMNS if c in schema.names],
        "casted": [
            c
            for c in DECIMAL_COLUMNS
            if c in schema.names and pa.types.is_decimal(schema.field(c).type)
        ],
    }


def process_file(src: Path, dst: Path, batch_size: int = BATCH_SIZE) -> int:
    """Потоково преобразует parquet-файл. Возвращает число строк."""
    reader = pq.ParquetFile(src)
    writer: pq.ParquetWriter | None = None
    rows = 0
    try:
        for batch in reader.iter_batches(batch_size=batch_size):
            table = transform(pa.Table.from_batches([batch]))
            if writer is None:
                writer = pq.ParquetWriter(
                    dst, table.schema, compression=COMPRESSION
                )
            writer.write_table(table)
            rows += table.num_rows
    finally:
        if writer is not None:
            writer.close()
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data-dir", type=Path, default=Path("dataset"),
        help="каталог с сырыми parquet-файлами (по умолчанию: dataset)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="куда писать результат (по умолчанию: <data-dir>/processed)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help=f"размер батча при потоковой обработке (по умолчанию: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="показать план изменений, ничего не записывая",
    )
    args = parser.parse_args(argv)

    data_dir: Path = args.data_dir
    out_dir: Path = args.out_dir or data_dir / "processed"

    missing = [n for n in DATASETS if not (data_dir / f"{n}.parquet").exists()]
    if missing:
        parser.error(
            f"в {data_dir} не найдены файлы: {', '.join(missing)}. "
            "Укажите верный --data-dir."
        )

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"вход:  {data_dir}")
    print(f"выход: {out_dir}{' (dry-run, запись отключена)' if args.dry_run else ''}\n")

    for name in DATASETS:
        src = data_dir / f"{name}.parquet"
        dst = out_dir / f"{name}.parquet"
        changes = describe_changes(src)

        print(f"{name}")
        print(f"  удалено колонок: {', '.join(changes['dropped']) or '—'}")
        print(f"  приведено к float64: {', '.join(changes['casted']) or '—'}")

        if args.dry_run:
            print(f"  строк: {pq.ParquetFile(src).metadata.num_rows:,}\n")
            continue

        rows = process_file(src, dst, batch_size=args.batch_size)
        size_mb = dst.stat().st_size / 1024 ** 2
        print(f"  строк: {rows:,}")
        print(f"  записано: {dst} ({size_mb:,.1f} МБ)\n")

    print("готово.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Предобработка датасета Search-Retrieval.

Приводит сырые parquet-файлы из `dataset/` к рабочему виду и складывает
результат в `dataset/processed/`.

Что делает:

1. Удаляет вырожденные колонки запроса (`search_category`,
   `search_is_delivery_search`) — см. обоснование в `notebooks/01_eda.ipynb`.
   Дроп применяется ко всем файлам, где эти колонки есть, включая `train`,
   чтобы набор признаков на обучении и на инференсе совпадал.
2. Приводит `decimal128`-колонки (`item_price`, `item_latitude`,
   `item_longitude`) к `float64` — в pandas они иначе приходят объектами
   `Decimal`, с которыми не работает ни арифметика, ни модели.
3. Разбирает `infm_params_text` модулем `infm_params`: добавляет фильтры
   запроса `filter_*`, поля объявления `item_*` и текст для ретрива
   `item_params_text` (см. `notebooks/02_infm_params.ipynb`).
4. Удаляет разобранные исходные тексты (`search_infm_params_text`,
   `item_infm_params_text`) и заменяет `item_category_id` флагом
   `item_is_service`.
5. Ставит колонки в порядке: сначала поля запроса (`query_id`, `search_*`,
   `filter_*`), затем поля объявления.

Исходные файлы в `dataset/` не меняются, поэтому при доработке словаря
ключей достаточно перезапустить скрипт.

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import infm_params as ip

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

#: Колонки, которые удаляются после разбора: их содержимое разложено
#: по `filter_*` / `item_*`.
DROP_AFTER_PARSING: tuple[str, ...] = (
    "search_infm_params_text",
    "item_infm_params_text",
)

#: `item_category_id` заменяется флагом: сам идентификатор в train равен 114
#: у 99.997% строк, но в корпусе 1 876 объявлений (1%) — не услуги, а товары
#: (категории 19, 112, 40, 33). Флаг сохраняет это различие.
SERVICE_CATEGORY_ID = 114
SERVICE_FLAG = "item_is_service"

#: Новые колонки и их типы. Типы заданы явно: иначе батч, где все списки
#: пустые, получил бы тип list<null> и схема поехала бы между батчами.
QUERY_FIELDS: dict[str, pa.DataType] = {
    "filter_category": pa.string(),
    "filter_subcategory": pa.list_(pa.string()),
    "filter_subject": pa.list_(pa.string()),
    "filter_online_booking": pa.bool_(),
}
ITEM_FIELDS: dict[str, pa.DataType] = {
    "item_category": pa.string(),
    "item_subcategory": pa.string(),
    "item_subjects": pa.list_(pa.string()),
    "item_online_booking": pa.bool_(),
    "item_params_text": pa.string(),
}

#: Префиксы колонок запроса — они идут первыми.
QUERY_PREFIXES: tuple[str, ...] = ("query_id", "search_", "filter_")

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


def add_service_flag(table: pa.Table) -> pa.Table:
    """Заменяет `item_category_id` булевым `item_is_service` на том же месте."""
    if "item_category_id" not in table.column_names:
        return table
    idx = table.column_names.index("item_category_id")
    is_service = pc.equal(table.column("item_category_id"), SERVICE_CATEGORY_ID)
    return table.set_column(idx, SERVICE_FLAG, is_service.cast(pa.bool_()))


def parse_params(table: pa.Table) -> pa.Table:
    """Добавляет колонки из `infm_params_text` и убирает разобранные тексты."""
    new: list[tuple[str, pa.Array]] = []
    for column, fields, parse in (
        ("search_infm_params_text", QUERY_FIELDS, ip.parse_search_filters),
        ("item_infm_params_text", ITEM_FIELDS, ip.parse_item_params),
    ):
        if column not in table.column_names:
            continue
        parsed = [parse(text) for text in table.column(column).to_pylist()]
        new += [(name, pa.array([row[name] for row in parsed], type=dtype))
                for name, dtype in fields.items()]

    present = [c for c in DROP_AFTER_PARSING if c in table.column_names]
    table = table.drop_columns(present) if present else table
    for name, values in new:
        table = table.append_column(name, values)
    return table


def order_columns(table: pa.Table) -> pa.Table:
    """Сначала поля запроса, затем поля объявления; внутри порядок сохраняется."""
    query = [c for c in table.column_names if c.startswith(QUERY_PREFIXES)]
    item = [c for c in table.column_names if c not in query]
    return table.select(query + item)


def transform(table: pa.Table) -> pa.Table:
    """Полный конвейер преобразований одного батча."""
    return order_columns(
        parse_params(add_service_flag(cast_decimals(drop_degenerate(table))))
    )


# --- ввод/вывод --------------------------------------------------------------


def describe_changes(src: Path) -> dict[str, list[str]]:
    """Что именно изменится в файле — считается по схеме, без чтения данных."""
    schema = pq.ParquetFile(src).schema_arrow
    added: list[str] = []
    if "item_category_id" in schema.names:
        added.append(f"{SERVICE_FLAG} (вместо item_category_id)")
    if "search_infm_params_text" in schema.names:
        added += list(QUERY_FIELDS)
    if "item_infm_params_text" in schema.names:
        added += list(ITEM_FIELDS)
    return {
        "dropped": [c for c in DROP_COLUMNS + DROP_AFTER_PARSING if c in schema.names],
        "casted": [
            c
            for c in DECIMAL_COLUMNS
            if c in schema.names and pa.types.is_decimal(schema.field(c).type)
        ],
        "added": added,
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
        print(f"  добавлено колонок: {', '.join(changes['added']) or '—'}")

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

"""Разбор полей `search_infm_params_text` и `item_infm_params_text`.

Оба поля — склеенные без разделителей пары «ключ значение» из формы Avito:

    Вид услуги Красота, здоровье Тип услуги СПА-услуги, массаж Онлайн-запись ...

Ключи у запроса и объявления общие, поэтому оба поля режутся одним
словарём ключей (`split_params`). Поверх разбора строятся:

Блок A — поля под фильтры запроса (выбраны в EDA по частоте в бенчмарке
и проверяемости по данным объявления):

    Вид услуги                 -> filter_category        / item_category
    Тип услуги                 -> filter_subcategory     / item_subcategory
    Предмет или специальность  -> filter_subject         / item_subjects
    Онлайн-запись              -> filter_online_booking  / item_online_booking

«Кто оказывает услуги» не используется: значения «Частный исполнитель» и
«Компания» по данным объявления не проверить, а «Женщина» / «Мужчина»
встречаются лишь в одном запросе бенчмарка.

Блок B — `item_params_text`: содержательная часть параметров для ретрива
(категории, названия услуг из прайс-листа, специальности, марки и т.п.)
без шаблонных полей, расписания, адреса и служебных признаков.

Вид и тип услуги с обеих сторон кодируются одним правилом:

    ключ со значением     -> значение
    ключ без значения     -> NO_CATEGORY («No category») / NO_SUBCATEGORY
    ключа нет             -> None

Ключ без значения — это отдельное состояние, а не пропуск: если в фильтре
запроса «Вид услуги» стоит без значения, 100% его позитивов — объявления с
тем же ключом без значения; для «Тип услуги» — 94.9% (у случайных объявлений
3.0%). Объявления, где ключа нет вовсе, среди этих позитивов почти не
встречаются, поэтому метка ставится только на ключ без значения.

`None` у запроса означает «фильтра нет», у объявления — «не проходит ни один
фильтр по этому полю». Восстановить пустой вид по `item_microcat_id` нельзя:
ни одна такая микрокатегория не встречается с заполненным видом.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Iterable

import pandas as pd

# --- словарь ключей -------------------------------------------------------------
# Группа определяет, какое значение ключ может принимать (см. `_accepts`)
# и попадает ли оно в текст для ретрива.

KEY_GROUPS: dict[str, tuple[str, ...]] = {
    # иерархия категорий
    "category": ("Вид услуги", "Тип услуги"),
    # содержательные поля: описывают саму услугу -> блок B
    "content": (
        "Тип услуги автосервиса", "Услуга", "Название услуги", "Услуги",
        "Специальность", "Специальность или сфера", "Предмет или специальность",
        "Специализация", "Чем вы занимаетесь", "Направление", "Мероприятия",
        "Что перевозите", "Груз", "Техника", "Тип техники", "Вид техники",
        "Тип землеройной техники", "Тип коммунальной техники",
        "Тип дорожно-строительной техники", "Марка", "Марка авто", "Модель",
        "Производители", "Что декорируете",
        "Какими комнатами занимаетесь", "Какими жилыми помещениями занимаетесь",
        "Какими нежилыми помещениями занимаетесь", "Транспорт", "Кузов",
        "Тип кузова", "Способ погрузки", "Дополнительно", "Вид товара", "Тип товара",
        "Тип подъёмной техники", "Чем занимается исполнитель",
    ),
    # атрибуты исполнителя и формата работы
    "attribute": (
        "Кто оказывает услуги", "Где вы оказываете услуги", "Ваши клиенты",
        "Как вы работаете", "Занятия", "Преподаватель", "Формат", "Аудитория",
        "Куда выезжаете", "Где проводите занятия", "Где снимаете", "Для кого",
        "Средства для уборки", "Камера наблюдения в ремонтной зоне",
        "Можно со своими запчастями", "Состояние", "Тип двигателя", "Доступность",
        "Кому подойдёт место", "Класс авто", "Коробка передач", "Руль",
        "Вид транспорта", "Тип помещения", "Тип автосервиса", "Способ оплаты",
        "Поездка", "Инструменты", "Гости", "Число исполнителей", "Загрузка",
        "Машины", "Вид объявления", "Учебное учреждение", "Место сделки",
        "Тип детали кузова", "Выезжаете ли к заказчику", "Проживание на объекте",
        "Страховка", "Тип аренды", "Как внести", "Перевозка", "Тип объявления",
    ),
    # булевы флаги: ключ без значения (иногда с «Да» / «Нет»)
    "flag": (
        "Онлайн-запись", "Онлайн-показ", "Гарантия", "Гарантия на работу",
        "Гарантия на выполнение работ", "Работа по договору", "Предоплата",
        "Готов закупить материалы", "Закупка материалов", "Берёте ли срочные заказы",
        "Бесплатная консультация", "Работаете с юрлицами и ИП", "Работаете с НДС",
        "Преподаёт носитель языка", "Работа в праздники и выходные", "Выезд",
        "Выезд к клиенту", "Выезд за город", "Выезд в день заказа", "Выезд и доставка",
        "Работа с техникой премиум-класса", "С оператором",
        "Своя музыкальная аппаратура", "Есть портфолио",
        "Готовность к командировкам", "Доставка", "Техническая поддержка сайта",
        "Подбираете ли студию", "Составление сценария", "Создание образа",
        "Печать фотокниги", "Детское кресло", "Дополнительный водитель",
        "Есть лимит пробега", "Цена с НДС", "Возможность выкупить авто",
    ),
    # числа: цены, опыт, объёмы
    "numeric": (
        "Стоимость", "Цена", "Начальная цена", "Тип стоимости", "Продолжительность",
        "Опыт работы", "Бригада", "Выполняю заказы", "Минимальная сумма заказа",
        "Минимальное время заказа", "Минимальное количество суток",
        "Грузоподъёмность", "Длина груза", "Ширина груза", "Число грузчиков",
        "Число мест", "Исполнителей в команде", "Площадь объекта", "Депозит",
        "Залог", "Год окончания", "Доплата за срочность", "Минимальное время аренды",
        "Расстояние доставки", "Срок выполнения заказа", "Минимальный возраст водителя",
        "Минимальный стаж вождения", "Через сколько дней возвращается", "Цена, ₽",
        "Сколько человек может участвовать", "Сколько гостей готовы снимать",
    ),
    # расписание
    "schedule": (
        "График работы", "График работы, дни недели", "Время работы,",
        "Время для связи", "Время для связи,", "Время для связи, дни недели",
        "Рабочие дни", "Дни",
    ),
    # адрес и гео
    "address": ("Место оказания услуг", "Метро", "Районы"),
    # служебные поля
    "meta": (
        "Признак предзаполнения прайс листа", "Признак мигрированного прайс листа",
        "Провайдер Календарей бронирования", "ID узла дерева навигации", "Видеофайлы",
    ),
    # встречаются только в фильтрах запроса
    "search": (
        "Рейтинг пользователя", "Срочная услуга (мультистатус)", "Поиск по слотам",
        "Сортировка для URL", "Участие в пилоте", "Открытие в сегменте",
        "Слова в описании", "Без депозита",
    ),
}

KEY_TO_GROUP: dict[str, str] = {
    key: group for group, keys in KEY_GROUPS.items() for key in keys
}

# индекс «первое слово ключа -> токены ключей», длинные ключи раньше коротких
_KEY_INDEX: dict[str, list[list[str]]] = defaultdict(list)
for _key in KEY_TO_GROUP:
    _KEY_INDEX[_key.split()[0]].append(_key.split())
for _candidates in _KEY_INDEX.values():
    _candidates.sort(key=len, reverse=True)

# строчные слова, с которых может начинаться значение числового поля или расписания
_LOWER_VALUE_WORDS = re.compile(
    r"^(от|до|с|за|больше|меньше|более|менее|пн|вт|ср|чт|пт|сб|вс)\.?$"
)
# ключи, чьё значение — свободный текст с любой буквы
_FREE_VALUE_KEYS = {"Слова в описании"}
# «Есть» сюда не входит: оно начинает флаги вроде «Есть портфолио»
_FLAG_VALUES = {"Да", "Нет"}

# значения-заглушки, бесполезные для ретрива
PLACEHOLDER_VALUES = {"", "Своя услуга", "Другое", "Другой", "Другая", "Свой"}

# ключ «Вид услуги» / «Тип услуги» есть, но без значения
NO_CATEGORY = "No category"
NO_SUBCATEGORY = "No subcategory"


def _accepts(key: str, next_token: str | None) -> bool:
    """Может ли `next_token` начинать значение ключа `key`.

    Защищает от ложных срабатываний, когда слово-ключ стоит внутри значения:
    «Вид услуги Доставка еды» — «Доставка» здесь не флаг, «Мероприятия Дни
    рождения» — «Дни» здесь не расписание.
    """
    if next_token is None or not next_token[0].isalpha() or next_token[0].isupper():
        return True
    group = KEY_TO_GROUP[key]
    if group in ("address", "meta") or key in _FREE_VALUE_KEYS:
        return True
    if group in ("numeric", "schedule"):
        return bool(_LOWER_VALUE_WORDS.match(next_token))
    return False


def split_params(text: str | None) -> list[tuple[str | None, str]]:
    """Режет infm_params_text на пары (ключ, значение).

    Ключ `None` означает текст, не привязанный ни к одному известному ключу
    (он встречается, если в словаре не хватает ключа).

    >>> split_params("Онлайн-запись Тип услуги СПА-услуги, массаж Вид услуги Красота, здоровье")
    [('Онлайн-запись', ''), ('Тип услуги', 'СПА-услуги, массаж'), ('Вид услуги', 'Красота, здоровье')]
    """
    if text is None or (isinstance(text, float) and text != text):
        return []
    tokens = str(text).split()
    pairs: list[tuple[str | None, str]] = []
    key: str | None = None
    value: list[str] = []
    i = 0

    def flush() -> None:
        if key is not None or value:
            pairs.append((key, " ".join(value)))

    while i < len(tokens):
        match = None
        for candidate in _KEY_INDEX.get(tokens[i], ()):
            n = len(candidate)
            if tokens[i:i + n] == candidate:
                name = " ".join(candidate)
                next_token = tokens[i + n] if i + n < len(tokens) else None
                if _accepts(name, next_token):
                    match = (name, n)
                    break
        if match is None:
            value.append(tokens[i])
            i += 1
            continue

        flush()
        name, n = match
        i += n
        if KEY_TO_GROUP[name] == "flag":
            # флаг не берёт значение, кроме явного «Да» / «Нет» / «Есть»
            flag_value = ""
            if i < len(tokens) and tokens[i] in _FLAG_VALUES:
                flag_value = tokens[i]
                i += 1
            pairs.append((name, flag_value))
            key, value = None, []
        else:
            key, value = name, []
    flush()
    return pairs


def _is_value(x) -> bool:
    """Непустая строка. В pandas пропуск бывает None или NaN (float)."""
    return isinstance(x, str) and x != ""


def _as_list(x) -> list[str]:
    """Скаляр, список или numpy-массив (после чтения parquet) -> список строк."""
    if isinstance(x, str):
        return [x] if x else []
    if x is None or isinstance(x, float):
        return []
    return [v for v in x if _is_value(v)]


def _unique(values: Iterable[str]) -> list[str]:
    """Уникальные значения с сохранением порядка, без учёта регистра."""
    seen, out = set(), []
    for v in values:
        norm = v.casefold()
        if v and norm not in seen:
            seen.add(norm)
            out.append(v)
    return out


def _flag_is_set(pairs: list[tuple[str | None, str]], name: str) -> bool:
    return any(k == name and v != "Нет" for k, v in pairs)


def _level_values(
    pairs: list[tuple[str | None, str]], key: str, empty_marker: str
) -> list[str]:
    """Значения уровня категории: значения / [empty_marker] / [] (ключа нет)."""
    values = _unique(v for k, v in pairs if k == key)
    if values:
        return values
    return [empty_marker] if any(k == key for k, _ in pairs) else []


# --- блок A: фильтры запроса ------------------------------------------------------

def parse_search_filters(text: str | None) -> dict:
    """Фильтры запроса, отобранные для поиска.

    Все остальные фильтры (Тип услуги автосервиса, Рейтинг пользователя,
    Кто оказывает услуги, Срочная услуга, Сортировка, служебные) отбрасываются.

    Ключ без значения -> `NO_CATEGORY` / `[NO_SUBCATEGORY]`, ключа нет ->
    `None` / `[]` (фильтра нет).
    """
    pairs = split_params(text)
    categories = _level_values(pairs, "Вид услуги", NO_CATEGORY)
    return {
        "filter_category": categories[0] if categories else None,
        "filter_subcategory": _level_values(pairs, "Тип услуги", NO_SUBCATEGORY),
        "filter_subject": _unique(v for k, v in pairs if k == "Предмет или специальность"),
        "filter_online_booking": _flag_is_set(pairs, "Онлайн-запись"),
    }


# --- блоки A и B: параметры объявления ---------------------------------------------

def params_text(pairs: list[tuple[str | None, str]]) -> str:
    """Блок B: содержательные значения для ретрива, через «; »."""
    values = (
        v for k, v in pairs
        if KEY_TO_GROUP.get(k) in ("category", "content") and v not in PLACEHOLDER_VALUES
    )
    return "; ".join(_unique(values))


def parse_item_params(text: str | None) -> dict:
    """Поля объявления под фильтры (блок A) и текст для ретрива (блок B).

    Ключ без значения -> `NO_CATEGORY` / `NO_SUBCATEGORY`, ключа нет -> `None`.
    """
    pairs = split_params(text)
    categories = _level_values(pairs, "Вид услуги", NO_CATEGORY)
    subcategories = _level_values(pairs, "Тип услуги", NO_SUBCATEGORY)
    return {
        "item_category": categories[0] if categories else None,
        "item_subcategory": subcategories[0] if subcategories else None,
        "item_subjects": _unique(v for k, v in pairs if k == "Предмет или специальность"),
        "item_online_booking": _flag_is_set(pairs, "Онлайн-запись"),
        "item_params_text": params_text(pairs),
    }


# --- обработка датафреймов ------------------------------------------------------------

def process_queries(df: pd.DataFrame, column: str = "search_infm_params_text") -> pd.DataFrame:
    """Добавляет колонки filter_* (блок A). Исходные колонки не меняются."""
    parsed = pd.DataFrame([parse_search_filters(t) for t in df[column]], index=df.index)
    return pd.concat([df, parsed], axis=1)


def process_items(df: pd.DataFrame, column: str = "item_infm_params_text") -> pd.DataFrame:
    """Добавляет колонки item_* (блоки A и B). Исходные колонки не меняются."""
    parsed = pd.DataFrame([parse_item_params(t) for t in df[column]], index=df.index)
    return pd.concat([df, parsed], axis=1)


# --- проверка фильтров ------------------------------------------------------------------

def match_filters(filters: dict, item: dict) -> dict[str, bool | None]:
    """Выполняет ли объявление каждый фильтр запроса: True / False / None.

    `None` — фильтр в запросе не задан. Пустое поле объявления (`None`)
    означает несоответствие, а `NO_CATEGORY` / `NO_SUBCATEGORY` совпадают
    только с такими же метками в фильтре. Так работает и фильтр самого Avito:
    объявления без вида услуги есть у 5.8% позитивов train без фильтра, но
    лишь у 0.15% позитивов с заданным видом услуги. Проверено и на качестве:
    строгая проверка теряет 0.1 п.п. позитивов по виду услуги и 0.5 п.п. по
    типу, зато пропускает 9.7% и 4.9% случайных объявлений против 13.9% и
    27.6% при мягкой.

    Несколько значений одного фильтра объединяются через ИЛИ (мультивыбор).
    Для жёсткой фильтрации объявление оставляют, если ни один фильтр не False.
    """
    def check(filter_value, item_value) -> bool | None:
        wanted = _as_list(filter_value)
        if not wanted:
            return None
        have = {v.casefold() for v in _as_list(item_value)}
        return any(w.casefold() in have for w in wanted)

    return {
        "category": check(filters["filter_category"], item["item_category"]),
        "subcategory": check(filters["filter_subcategory"], item["item_subcategory"]),
        "subject": check(filters["filter_subject"], item["item_subjects"]),
        "online_booking": (bool(item["item_online_booking"])
                           if filters["filter_online_booking"] else None),
    }

"""Dense-ретриверы: эмбеддинги каталога с кэшем и точный поиск на GPU.

Модели используются без дообучения, с префиксами, с которыми их обучали
авторы (`MODELS`). Эмбеддинги нормируются, так что скалярное произведение —
косинус. Поиск точный: матричное умножение на GPU, без приближённого индекса.

Эмбеддинги каталога считаются долго (около часа на модель на ноутбучной GPU:
~60 документов/с у e5-large), поэтому кэшируются частями: каталог валидации — в
`dataset/embeddings/<модель>/`, корпус бенчмарка — в
`dataset/embeddings/benchmark/<модель>/`. Прерванный запуск продолжается с
последней готовой части. Рядом лежит список `item_id`: если корпус изменился
(другое разбиение), кэш пересчитывается.
Тексты документов — `texts.doc_texts`; при изменении чистки кэш надо удалить.

Модели видят первые 512 токенов: на выборке корпуса длиннее 512 токенов 28%
документов у e5-large и 24% у RoSBERTa (медиана — ~300 токенов), обрезается
хвост описания.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from paths import EMBEDDINGS

#: Корпуса: каталог валидации (dataset/split), корпус бенчмарка (benchmark_items) и
#: объявления train вне каталога (для обучения на всём train; data.load_train_extra_items).
CATALOG, BENCHMARK, TRAIN_EXTRA = "catalog", "benchmark", "train_extra"

PART_SIZE = 16_384
PROGRESS_STEP = 2_048  # шаг прогресс-бара при кодировании каталога
BATCH_SIZE = 64


@dataclass(frozen=True)
class DenseModel:
    hf_id: str
    query_prefix: str
    doc_prefix: str


#: Модели и их префиксы — так, как их обучали авторы.
MODELS: dict[str, DenseModel] = {
    "e5-large": DenseModel("intfloat/multilingual-e5-large", "query: ", "passage: "),
    "RoSBERTa": DenseModel("ai-forever/ru-en-RoSBERTa", "search_query: ", "search_document: "),
}


def cache_dir(name: str, corpus: str = CATALOG):
    """Каталог валидации — dataset/embeddings/<модель>, бенчмарк — .../benchmark/<модель>."""
    return EMBEDDINGS / name if corpus == CATALOG else EMBEDDINGS / corpus / name


def load_model(name: str):
    import torch
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODELS[name].hf_id, device="cuda",
                               model_kwargs={"dtype": torch.float16})


def encode(model, texts: list[str], prefix: str, batch_size: int = BATCH_SIZE,
           progress: bool = False) -> np.ndarray:
    """Нормированные эмбеддинги float16. `progress=True` — прогресс-бар по батчам."""
    return model.encode([prefix + t for t in texts], batch_size=batch_size,
                        normalize_embeddings=True, convert_to_numpy=True,
                        show_progress_bar=progress).astype(np.float16)


def catalog_embeddings(name: str, item_ids, docs: list[str] | None = None,
                       model=None, verbose: bool = True,
                       corpus: str = CATALOG) -> np.ndarray:
    """Эмбеддинги корпуса из кэша; недостающие части досчитываются (нужны docs).

    `corpus` — какой корпус кодируется: каталог валидации или корпус бенчмарка.
    У каждого свой кэш, чтобы они не затирали друг друга.
    """
    item_ids = np.asarray(item_ids).astype(str)
    folder = cache_dir(name, corpus)
    ids_path = folder / "item_ids.npy"
    if ids_path.exists() and not np.array_equal(np.load(ids_path), item_ids):
        if verbose:
            print(f"{name}: каталог изменился, кэш пересчитывается")
        for f in folder.glob("part_*.npy"):
            f.unlink()
    folder.mkdir(parents=True, exist_ok=True)
    np.save(ids_path, item_ids)

    n_parts = (len(item_ids) + PART_SIZE - 1) // PART_SIZE
    missing = [i for i in range(n_parts) if not (folder / f"part_{i:03d}.npy").exists()]
    if missing:
        if docs is None:
            raise ValueError(f"{name}: в кэше нет {len(missing)} частей, нужны тексты docs")
        from tqdm.auto import tqdm

        model = model or load_model(name)
        parts = {i: docs[i * PART_SIZE:(i + 1) * PART_SIZE] for i in missing}
        bar = tqdm(total=sum(map(len, parts.values())), unit="док", disable=not verbose,
                   desc=f"{name}: каталог, частей в кэше {n_parts - len(missing)}/{n_parts}")
        # часть кодируется кусками, чтобы бар двигался чаще, чем раз в несколько минут
        for done, (i, texts) in enumerate(parts.items(), start=n_parts - len(missing) + 1):
            chunks = []
            for j in range(0, len(texts), PROGRESS_STEP):
                chunks.append(encode(model, texts[j:j + PROGRESS_STEP], MODELS[name].doc_prefix))
                bar.update(len(chunks[-1]))
            np.save(folder / f"part_{i:03d}.npy", np.concatenate(chunks))
            bar.set_description(f"{name}: каталог, частей в кэше {done}/{n_parts}")
        bar.close()
    return np.concatenate([np.load(folder / f"part_{i:03d}.npy") for i in range(n_parts)])


class DenseRetriever:
    """Ретривер по косинусу; эмбеддинги каталога и запросов лежат на GPU."""

    def __init__(self, name: str, item_ids, docs: list[str] | None = None,
                 corpus: str = CATALOG):
        import torch

        self.name = name
        self._doc_emb = torch.from_numpy(
            catalog_embeddings(name, item_ids, docs, corpus=corpus)).cuda()
        self._query_emb = None

    # интерфейс ретривера (см. retrieve.py)
    def prepare(self, texts: np.ndarray) -> None:
        """Кодирует запросы; модель нужна только на это время."""
        import gc

        import torch

        model = load_model(self.name)
        self._query_emb = torch.from_numpy(
            encode(model, list(texts), MODELS[self.name].query_prefix, progress=True)).cuda()
        del model
        gc.collect()
        torch.cuda.empty_cache()

    def scores(self, rows: slice) -> np.ndarray:
        return (self._query_emb[rows] @ self._doc_emb.T).float().cpu().numpy()

    def pair_scores(self, query_rows: np.ndarray, items: np.ndarray,
                    batch: int = 100_000) -> np.ndarray:
        """Косинус отдельных пар: запрос `texts[query_rows[i]]` — документ `items[i]`."""
        import torch

        out = np.empty(len(items), dtype=np.float32)
        for s in range(0, len(items), batch):
            q = self._query_emb[torch.from_numpy(query_rows[s:s + batch]).cuda()]
            d = self._doc_emb[torch.from_numpy(items[s:s + batch]).cuda()]
            out[s:s + batch] = (q * d).sum(dim=1).float().cpu().numpy()
        return out

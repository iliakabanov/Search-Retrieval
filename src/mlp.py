"""MLP-переранжировщик с lookup-эмбеддингами (docs/reranker_plan.md, модель 2).

Вход пары «запрос — кандидат»:

* числовые признаки из features.py (те же, что у LightGBM), нормированные
  `Preprocessor`: пропуски -> медиана + флаг «было пусто», тяжёлые хвосты -> log,
  затем стандартизация и обрезка до ±5;
* обучаемые эмбеддинги категорий: микрокатегория и вид услуги объявления, вид
  услуги из фильтра запроса, локация объявления и локация запроса. Отдельно —
  «сродство» локаций: скалярное произведение эмбеддингов локации запроса и
  локации объявления, чтобы модель сама выучила, какие локации близки (регион —
  его города, город — соседние).

Эмбеддинг конкретного объявления не используется: на бенчмарке 90% объявлений
не встречаются в train (см. README, «Переранжировщик»).

Обучение — listwise: softmax по кандидатам события, цель — позитивы события
(равными долями).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

#: Признаки с тяжёлым хвостом — логарифмируются перед стандартизацией.
LOG_FEATURES = ("geo_dist_km",)
#: Категориальные входы и размерности их эмбеддингов.
EMBEDDINGS = {"microcat": 16, "category": 8, "query_category": 8, "item_loc": 16, "query_loc": 16}
AFFINITY_DIM = 16


class Preprocessor:
    """Нормировка числовых признаков; параметры считаются по обучению."""

    def __init__(self, columns: list[str]):
        self.columns = list(columns)
        self.stats: dict[str, dict] = {}

    def _raw(self, df: pd.DataFrame, c: str) -> np.ndarray:
        x = df[c].to_numpy(dtype=np.float32)
        return np.log1p(np.clip(x, 0, None)) if c in LOG_FEATURES else x

    def fit(self, df: pd.DataFrame, sample: int = 2_000_000, seed: int = 0) -> "Preprocessor":
        # статистик по 2 млн случайных строк достаточно, а на всех 17 млн это минуты
        if len(df) > sample:
            df = df.iloc[np.random.default_rng(seed).choice(len(df), sample, replace=False)]
        for c in self.columns:
            x = self._raw(df, c).astype(np.float64)
            nan = np.isnan(x)
            med = float(np.nanmedian(x)) if (~nan).any() else 0.0
            x = np.where(nan, med, x)
            self.stats[c] = {"median": med, "mean": float(x.mean()),
                             "std": float(x.std()) or 1.0, "has_nan": bool(nan.any())}
        return self

    @property
    def n_out(self) -> int:
        return len(self.columns) + sum(s["has_nan"] for s in self.stats.values())

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        # пишем сразу в float16 по колонке: весь массив во float64 — это гигабайты
        out = np.empty((len(df), self.n_out), dtype=np.float16)
        j = 0
        for c in self.columns:
            s = self.stats[c]
            x = self._raw(df, c)
            nan = np.isnan(x)
            out[:, j] = np.clip((np.where(nan, s["median"], x) - s["mean"]) / s["std"], -5, 5)
            j += 1
            if s["has_nan"]:
                out[:, j] = nan
                j += 1
        return out

    def to_json(self) -> dict:
        return {"columns": self.columns, "stats": self.stats}

    @classmethod
    def from_json(cls, d: dict) -> "Preprocessor":
        p = cls(d["columns"])
        p.stats = d["stats"]
        return p


class Vocab:
    """Значение категории -> индекс эмбеддинга; 0 — неизвестное значение.

    Значения — числа (микрокатегория, локация) или строки (вид услуги); кодируются
    через pd.Index без перевода в строки: на 17 млн значений это секунды, а не минуты.
    """

    def __init__(self, values):
        self.index = pd.Index(pd.unique(np.asarray(values)))

    def __len__(self) -> int:
        return len(self.index) + 1

    def encode(self, values) -> np.ndarray:
        pos = self.index.get_indexer(np.asarray(values))
        return (pos + 1).astype(np.int32)       # -1 (нет в словаре) -> 0

    def to_json(self) -> list:
        return [v.item() if hasattr(v, "item") else v for v in self.index]

    @classmethod
    def from_json(cls, values: list[str]) -> "Vocab":
        return cls(values)


class RerankMLP(nn.Module):
    """`text_dim > 0` — текстовая часть: векторы запроса и объявления (e5, 1024 числа)
    сжимаются линейными слоями до `text_dim` и подаются в сеть как [q, d, q⊙d].
    Сжатие и dropout ограничивают запоминание конкретных объявлений по их векторам."""

    def __init__(self, n_num: int, vocab_sizes: dict[str, int], hidden=(256, 128),
                 dropout: float = 0.1, text_dim: int = 0, text_in: int = 1024,
                 text_dropout: float = 0.2):
        super().__init__()
        self.text_dim = text_dim
        if text_dim:
            self.proj_q = nn.Linear(text_in, text_dim)
            self.proj_d = nn.Linear(text_in, text_dim)
            self.text_drop = nn.Dropout(text_dropout)
        self.emb = nn.ModuleDict({k: nn.Embedding(vocab_sizes[k], d) for k, d in EMBEDDINGS.items()})
        self.aff_query = nn.Embedding(vocab_sizes["query_loc"], AFFINITY_DIM)
        self.aff_item = nn.Embedding(vocab_sizes["item_loc"], AFFINITY_DIM)
        layers, d_in = [], n_num + sum(EMBEDDINGS.values()) + 1 + 3 * text_dim
        for h in hidden:
            layers += [nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(dropout)]
            d_in = h
        layers.append(nn.Linear(d_in, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, num: torch.Tensor, cats: dict[str, torch.Tensor],
                q_vec: torch.Tensor | None = None, d_vec: torch.Tensor | None = None
                ) -> torch.Tensor:
        affinity = (self.aff_query(cats["query_loc"]) * self.aff_item(cats["item_loc"])).sum(-1)
        parts = [num] + [self.emb[k](cats[k]) for k in EMBEDDINGS] + [affinity.unsqueeze(-1)]
        if self.text_dim:
            q = self.text_drop(self.proj_q(q_vec.float()))
            d = self.text_drop(self.proj_d(d_vec.float()))
            parts += [q, d, q * d]
        x = torch.cat(parts, dim=-1)
        return self.mlp(x).squeeze(-1)


def listwise_loss(scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax по кандидатам группы; цель — позитивы группы равными долями.

    scores, labels, mask: [группы, кандидаты]; mask — настоящие (не паддинг) кандидаты.
    """
    scores = scores.masked_fill(~mask, float("-inf"))
    log_p = torch.log_softmax(scores, dim=1)
    target = labels / labels.sum(dim=1, keepdim=True).clamp(min=1)
    return -(target * log_p.masked_fill(~mask, 0)).sum(dim=1).mean()

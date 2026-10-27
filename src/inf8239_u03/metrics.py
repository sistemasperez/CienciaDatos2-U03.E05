from __future__ import annotations

import numpy as np
import pandas as pd


def hit_rate_at_k(recommendations: dict[int, list[int]], truth: pd.DataFrame, k: int = 10) -> float:
    hits = []
    for row in truth.itertuples():
        if row.userId in recommendations:
            hits.append(int(row.movieId in recommendations[row.userId][:k]))
    return float(np.mean(hits)) if hits else 0.0


def precision_at_k(recommendations: dict[int, list[int]], truth: pd.DataFrame, k: int = 10) -> float:
    user_truth = truth.groupby("userId")["movieId"].apply(set).to_dict()
    scores = []
    for user_id, true_items in user_truth.items():
        if user_id in recommendations:
            preds = recommendations[user_id][:k]
            hits = len(set(preds) & true_items)
            scores.append(hits / k if k > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def recall_at_k(recommendations: dict[int, list[int]], truth: pd.DataFrame, k: int = 10) -> float:
    user_truth = truth.groupby("userId")["movieId"].apply(set).to_dict()
    scores = []
    for user_id, true_items in user_truth.items():
        if user_id in recommendations:
            preds = recommendations[user_id][:k]
            hits = len(set(preds) & true_items)
            scores.append(hits / len(true_items) if true_items else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def ndcg_at_k(recommendations: dict[int, list[int]], truth: pd.DataFrame, k: int = 10) -> float:
    user_truth = truth.groupby("userId")["movieId"].apply(set).to_dict()
    scores = []
    for user_id, true_items in user_truth.items():
        if user_id in recommendations:
            preds = recommendations[user_id][:k]
            dcg = 0.0
            for rank, item in enumerate(preds, start=1):
                if item in true_items:
                    dcg += 1.0 / np.log2(rank + 1)
            n_relevant = min(k, len(true_items))
            idcg = sum(1.0 / np.log2(r + 1) for r in range(1, n_relevant + 1))
            scores.append(dcg / idcg if idcg > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def catalog_coverage(recommendations: dict[int, list[int]], catalog_size: int) -> float:
    recommended = {item for values in recommendations.values() for item in values}
    return len(recommended) / catalog_size if catalog_size else 0.0

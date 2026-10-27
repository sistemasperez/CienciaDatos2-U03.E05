import numpy as np
import pandas as pd
import pytest

from inf8239_u03.metrics import (
    catalog_coverage,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_ranking_metrics_baseline():
    recs = {1: [3, 2], 2: [4, 1]}
    truth = pd.DataFrame({"userId": [1, 2], "movieId": [3, 5]})
    assert hit_rate_at_k(recs, truth, 2) == 0.5
    assert catalog_coverage(recs, 5) == 0.8


def test_ranking_metrics_precision_recall_ndcg():
    recs = {1: [3, 2], 2: [4, 1]}
    truth = pd.DataFrame({"userId": [1, 2], "movieId": [3, 5]})
    # User 1: top 2 has item 3 at rank 1 (1 hit). precision=1/2=0.5, recall=1/1=1.0, ndcg=1.0
    # User 2: top 2 has no match (0 hits). precision=0.0, recall=0.0, ndcg=0.0
    assert precision_at_k(recs, truth, 2) == 0.25
    assert recall_at_k(recs, truth, 2) == 0.50
    assert ndcg_at_k(recs, truth, 2) == 0.50


def test_ndcg_discount():
    # User 1 has target item at rank 2
    recs = {1: [99, 3]}
    truth = pd.DataFrame({"userId": [1], "movieId": [3]})
    expected_ndcg = (1.0 / np.log2(2 + 1)) / (1.0 / np.log2(1 + 1))  # 1/log2(3) approx 0.6309
    assert np.isclose(ndcg_at_k(recs, truth, 2), expected_ndcg)

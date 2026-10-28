import numpy as np
import pandas as pd
import pytest

from movie_recs.model import fit_model, ranking_metrics, recommend, temporal_split


@pytest.fixture
def model():
    return {
        "mean": np.array(3.0),
        "bu": np.zeros(3),
        "bi": np.zeros(6),
        "P": np.ones((3, 2)),
        "Q": np.ones((6, 2)),
        "counts": np.array([0, 9, 8, 7, 6, 0]),
        "seen": np.array([[0] * 6, [0, 1, 0, 0, 0, 0], [0] * 6], dtype=bool),
    }


def test_ranking_excludes_seen_unavailable_and_explicitly_excluded_items(model):
    assert recommend(model, 1, exclude=[2]).tolist() == [3, 4]


def test_cold_user_receives_deterministic_popularity_fallback(model):
    assert recommend(model, 99).tolist() == [1, 2, 3, 4]


def test_all_user_evaluation_includes_cold_user_fallback(model):
    prior = pd.DataFrame({"user": [1], "item": [1]})
    heldout = pd.DataFrame({"user": [99], "item": [1], "rating": [5]})
    assert ranking_metrics(model, heldout, prior, "factorization")["evaluated_users"] == 0
    result = ranking_metrics(model, heldout, prior, "factorization", include_cold=True)
    assert result["evaluated_users"] == 1
    assert result["ndcg_at_10"] == 1.0


def test_ranking_recall_and_ndcg_for_known_relevance(model):
    prior = pd.DataFrame({"user": [1], "item": [1]})
    heldout = pd.DataFrame({"user": [1], "item": [2], "rating": [5]})
    result = ranking_metrics(model, heldout, prior, "popularity")
    assert result["recall_at_10"] == result["ndcg_at_10"] == 1.0
    assert result["evaluated_users"] == 1


def test_global_time_split_keeps_timestamp_ties_together():
    frame = pd.DataFrame(
        {"user": 1, "item": np.arange(100), "rating": 4, "timestamp": np.arange(100) // 2}
    )
    fit, valid, test = temporal_split(frame)
    assert (
        fit.timestamp.max() < valid.timestamp.min() <= valid.timestamp.max() < test.timestamp.min()
    )
    assert len(fit) + len(valid) + len(test) == 100


def test_matrix_factorization_learns_observed_ratings_without_nan():
    frame = pd.DataFrame(
        {
            "user": [1, 1, 2, 2, 3, 3],
            "item": [1, 2, 1, 3, 2, 3],
            "rating": [5.0, 4.0, 4.0, 1.0, 3.0, 2.0],
        }
    )
    fitted, _ = fit_model(frame, factors=2, regularization=0.1, maxiter=30)
    assert np.isfinite(fitted["P"]).all()
    assert fitted["seen"].sum() == 6


def test_invalid_recommendation_limit(model):
    with pytest.raises(ValueError):
        recommend(model, 1, k=0)

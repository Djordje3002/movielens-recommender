"""Biased matrix factorization with global chronological evaluation."""

import io
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .common import archive, figure, regression, save_report


def temporal_split(frame):
    frame = frame.sort_values(["timestamp", "user", "item"]).reset_index(drop=True)
    # Timestamp quantiles with strict boundaries keep ties in the same partition.
    first, second = frame.timestamp.quantile([0.8, 0.9]).to_numpy()
    return (
        frame[frame.timestamp <= first],
        frame[(frame.timestamp > first) & (frame.timestamp <= second)],
        frame[frame.timestamp > second],
    )


def fit_model(frame, factors=16, regularization=10.0, maxiter=100):
    users, items = int(frame.user.max()) + 1, int(frame.item.max()) + 1
    u, i, rating = frame.user.to_numpy(), frame.item.to_numpy(), frame.rating.to_numpy()
    mean = float(rating.mean())
    rng = np.random.default_rng(42)
    start = np.r_[np.zeros(users + items), rng.normal(0, 0.05, (users + items) * factors)]

    def unpack(vector):
        offset = users + items
        return (
            vector[:users],
            vector[users:offset],
            vector[offset : offset + users * factors].reshape(users, factors),
            vector[offset + users * factors :].reshape(items, factors),
        )

    def objective(vector):
        bu, bi, P, Q = unpack(vector)
        error = mean + bu[u] + bi[i] + np.sum(P[u] * Q[i], axis=1) - rating
        value = (np.dot(error, error) + regularization * np.dot(vector, vector)) / len(frame)
        dbu = np.bincount(u, weights=error, minlength=users)
        dbi = np.bincount(i, weights=error, minlength=items)
        dP, dQ = np.empty_like(P), np.empty_like(Q)
        for feature in range(factors):
            dP[:, feature] = np.bincount(u, weights=error * Q[i, feature], minlength=users)
            dQ[:, feature] = np.bincount(i, weights=error * P[u, feature], minlength=items)
        gradient = (
            2 * (np.r_[dbu, dbi, dP.ravel(), dQ.ravel()] + regularization * vector) / len(frame)
        )
        return float(value), gradient

    solution = minimize(
        objective, start, method="L-BFGS-B", jac=True, options={"maxiter": maxiter, "ftol": 1e-7}
    )
    bu, bi, P, Q = unpack(solution.x)
    counts = np.bincount(i, minlength=items)
    seen = np.zeros((users, items), dtype=bool)
    seen[u, i] = True
    return {
        "mean": np.array(mean),
        "bu": bu,
        "bi": bi,
        "P": P,
        "Q": Q,
        "counts": counts,
        "seen": seen,
    }, {
        "iterations": int(solution.nit),
        "converged": bool(solution.success),
        "message": str(solution.message),
    }


def predict_ratings(model, frame):
    u, i = frame.user.to_numpy(), frame.item.to_numpy()
    value = (
        float(model["mean"])
        + model["bu"][u]
        + model["bi"][i]
        + np.sum(model["P"][u] * model["Q"][i], axis=1)
    )
    return np.clip(value, 1, 5)


def recommend(model, user, k=10, exclude=(), method="factorization"):
    if not 1 <= k <= 100:
        raise ValueError("k must be in 1..100")
    catalog = np.flatnonzero(model["counts"] > 0)
    warm = 0 <= user < len(model["bu"]) and model["seen"][user].any()
    if method == "popularity" or not warm:
        scores = model["counts"][catalog].astype(float)
    else:
        scores = (
            float(model["mean"])
            + model["bu"][user]
            + model["bi"][catalog]
            + model["Q"][catalog] @ model["P"][user]
        )
    consumed = set(exclude)
    if warm:
        consumed.update(np.flatnonzero(model["seen"][user]))
    keep = ~np.isin(catalog, list(consumed))
    catalog, scores = catalog[keep], scores[keep]
    order = np.lexsort((catalog, -scores))[:k]
    return catalog[order]


def ranking_metrics(model, heldout, prior, method, include_cold=False):
    recalls, ndcgs, covered = [], [], set()
    for user, rows in heldout.groupby("user"):
        if not include_cold and (user >= len(model["bu"]) or not model["seen"][user].any()):
            continue
        excluded = prior.loc[prior.user == user, "item"].to_numpy()
        relevant = set(rows.loc[rows.rating >= 4, "item"]) - set(excluded)
        relevant = {i for i in relevant if i < len(model["counts"]) and model["counts"][i] > 0}
        if not relevant:
            continue
        top = recommend(model, int(user), exclude=excluded, method=method)
        hits = np.array([item in relevant for item in top], dtype=float)
        recalls.append(float(hits.sum() / len(relevant)))
        ideal = np.sum(1 / np.log2(np.arange(min(len(relevant), 10)) + 2))
        ndcgs.append(float(np.sum(hits / np.log2(np.arange(len(top)) + 2)) / ideal))
        covered.update(top)
    return {
        "recall_at_10": float(np.mean(recalls)) if recalls else 0.0,
        "ndcg_at_10": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "evaluated_users": len(recalls),
        "catalog_coverage_at_10": len(covered) / max(1, int((model["counts"] > 0).sum())),
    }


def warm_rows(model, frame):
    rows = frame[(frame.user < len(model["bu"])) & (frame.item < len(model["bi"]))]
    return rows[
        model["seen"][rows.user.to_numpy()].any(axis=1)
        & (model["counts"][rows.item.to_numpy()] > 0)
    ]


def train(root):
    with archive(root) as source:
        frame = pd.read_csv(
            io.BytesIO(source.read("ml-100k/u.data")),
            sep="\t",
            names=["user", "item", "rating", "timestamp"],
        )
    fit, valid, test = temporal_split(frame)
    model, optimization = fit_model(fit)
    validation = {
        method: ranking_metrics(model, valid, fit, method)
        for method in ["popularity", "factorization"]
    }
    selected = max(validation, key=lambda name: validation[name]["ndcg_at_10"])
    prior = pd.concat([fit, valid])
    ranking = {
        method: ranking_metrics(model, test, prior, method)
        for method in ["popularity", "factorization"]
    }
    all_user_ranking = {
        method: ranking_metrics(model, test, prior, method, include_cold=True)
        for method in ["popularity", "factorization"]
    }
    warm = warm_rows(model, test)
    rating_results = {
        "global_mean_baseline": regression(warm.rating, np.full(len(warm), float(model["mean"]))),
        "factorization": regression(warm.rating, predict_ratings(model, warm)),
    }
    _, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, key in zip(axes, ["recall_at_10", "ndcg_at_10"]):
        ax.bar(
            list(ranking), [value[key] for value in ranking.values()], color=["#94a3b8", "#2563eb"]
        )
        ax.set_title(key)
    figure(root, "evaluation.png")
    result = save_report(
        root,
        {
            "raw_rows": len(frame),
            "split": {
                "train": len(fit),
                "validation": len(valid),
                "test": len(test),
                "method": "global timestamp 80/10/10; timestamp ties kept together",
            },
            "selected": selected,
            "optimization": optimization,
            "validation": validation,
            "test_ranking": ranking,
            "all_user_test_ranking": all_user_ranking,
            "test_ratings": rating_results,
            "warm_test_ratings": len(warm),
            "excluded_cold_test_ratings": len(test) - len(warm),
            "catalog_size": int((model["counts"] > 0).sum()),
        },
    )
    np.savez_compressed(root / "artifacts/model.npz", **model)
    (root / "artifacts/selection.json").write_text(json.dumps({"method": selected}))
    return result


def predict(root, payload):
    user = payload["user_id"]
    if not isinstance(user, int) or isinstance(user, bool) or user < 1:
        raise ValueError("user_id must be a positive integer")
    method = json.loads((root / "artifacts/selection.json").read_text())["method"]
    with np.load(root / "artifacts/model.npz", allow_pickle=False) as model:
        top = recommend(model, user, k=int(payload.get("k", 10)), method=method)
    return {
        "user_id": user,
        "movie_ids": top.tolist(),
        "method": method,
        "note": "Unseen users receive the training-popularity fallback",
    }

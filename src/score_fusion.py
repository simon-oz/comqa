"""Combine heterogeneous classifier scores using ranks fitted on dev rows only."""

import numpy as np
from sklearn.metrics import average_precision_score


class RankScoreFusion:
    """Saved, reusable score transform; inputs are aligned (rows, labels) arrays."""

    def __init__(self, weights=None, weight_trials=0, seed=42):
        self.weights = weights
        self.weight_trials = weight_trials
        self.seed = seed

    def fit(self, scores, labels):
        self.members_ = list(scores)
        if len(self.members_) < 2:
            raise ValueError("Score fusion needs at least two members")
        self.reference_ = {}
        for name, values in scores.items():
            values = np.asarray(values, dtype=float)
            if values.shape != labels.shape or not np.isfinite(values).all():
                raise ValueError(f"Invalid or unaligned dev scores for {name}")
            self.reference_[name] = np.sort(values, axis=0)
        weights = self.weights or {name: 1.0 for name in self.members_}
        if set(weights) != set(self.members_):
            raise ValueError("Weights must name every fusion member exactly once")
        values = np.array([weights[name] for name in self.members_], dtype=float)
        if not np.isfinite(values).all() or (values < 0).any() or values.sum() <= 0:
            raise ValueError("Fusion weights must be finite, nonnegative, and have positive sum")
        values /= values.sum()
        transformed = np.stack([self._rank(name, scores[name]) for name in self.members_])
        candidates = [values]
        if self.weight_trials:
            rng = np.random.default_rng(self.seed)
            candidates.extend(rng.dirichlet(np.ones(len(values)) * 2, self.weight_trials))
        best = -np.inf
        for candidate in candidates:
            combined = np.tensordot(candidate, transformed, axes=(0, 0))
            score = float(average_precision_score(labels, combined, average="macro"))
            if score > best:
                best, values = score, candidate
        self.weights_ = dict(zip(self.members_, values.tolist()))
        return self

    def _rank(self, name, values):
        values = np.asarray(values, dtype=float)
        reference = self.reference_[name]
        if values.ndim != 2 or values.shape[1] != reference.shape[1] or not np.isfinite(values).all():
            raise ValueError(f"Invalid scores for {name}")
        ranks = np.empty_like(values)
        for label in range(reference.shape[1]):
            column = reference[:, label]
            # Midranks preserve ties, including hard 0/1 Qwen outputs.
            left = np.searchsorted(column, values[:, label], side="left")
            right = np.searchsorted(column, values[:, label], side="right")
            ranks[:, label] = (left + right) / (2 * len(column))
        return ranks

    def transform(self, scores):
        if set(scores) != set(self.members_):
            raise ValueError("Fusion member set differs from the fitted member set")
        transformed = [self._rank(name, scores[name]) for name in self.members_]
        if len({array.shape for array in transformed}) != 1:
            raise ValueError("Fusion members must have aligned rows and labels")
        return sum(self.weights_[name] * array for name, array in zip(self.members_, transformed))

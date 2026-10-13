"""Shared embedding classification helpers (also the import path for saved models)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    hamming_loss, precision_recall_curve,
)
from sklearn.model_selection import train_test_split

TEXT_COL = "question"
LABEL_COLS = [
    "retrieval", "comparison", "aggregation", "context_resolution", "temporal",
    "legal_citation", "evidence_extraction", "multi_document", "summary",
    "analyze", "evaluate", "recommend",
]
BASELINE_MODEL = "intfloat/e5-large-v2"
# These are classification features: BGE retrieval instructions do not apply.
MODEL_PROMPTS = {
    BASELINE_MODEL: "query: ",
    "sentence-transformers/all-MiniLM-L6-v2": "",
    "intfloat/e5-base-v2": "query: ",
    "sentence-transformers/sentence-t5-base": "",
    "sentence-transformers/sentence-t5-large": "",
    "google/embeddinggemma-300m": "task: classification | query: ",
    "BAAI/bge-m3": "",
    "BAAI/bge-base-en-v1.5": "",
    "BAAI/bge-large-en": "",
}


def load_dataset(path, *, require_labels):
    """Reject incomplete labels instead of silently treating unknown labels as 0."""
    records = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            text = record.get(TEXT_COL)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{path}:{line_number}: question must be nonempty text")
            records.append(record)
    if not records:
        raise ValueError(f"Dataset is empty: {path}")
    df = pd.DataFrame(records)
    present = [label for label in LABEL_COLS if label in df.columns]
    if require_labels or present:
        missing = sorted(set(LABEL_COLS) - set(present))
        if missing:
            raise ValueError(f"{path}: missing label columns: {', '.join(missing)}")
        for label in LABEL_COLS:
            values = pd.to_numeric(df[label], errors="coerce")
            if not np.isfinite(values.to_numpy(dtype=float)).all():
                raise ValueError(f"{path}: label {label!r} has missing or invalid values")
            # Preserve the existing convention: values such as retrieval=2 mean 1.
            df[label] = (values > 0).astype(int)
    return df, bool(present)


def split_indices(df, dev_size, seed):
    indices = np.arange(len(df))
    combos = ["|".join(map(str, row)) for row in df[LABEL_COLS].to_numpy()]
    try:
        train, dev = train_test_split(
            indices, test_size=dev_size, random_state=seed, stratify=combos,
        )
        strategy = "label_combination_stratified"
    except ValueError:
        train, dev = train_test_split(indices, test_size=dev_size, random_state=seed)
        strategy = "random_fallback"
    return train, dev, strategy


class SafeOneVsRestClassifier:
    """Train one binary classifier per label, including constant training labels."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimators_ = []
        self.constant_labels_ = []
        for col in y.T:
            unique = np.unique(col)
            constant = int(unique[0]) if len(unique) == 1 else None
            estimator = None if constant is not None else clone(self.estimator).fit(X, col)
            self.estimators_.append(estimator)
            self.constant_labels_.append(constant)
        return self

    def scores(self, X, prefer_proba=False):
        columns = []
        for estimator, constant in zip(self.estimators_, self.constant_labels_):
            if estimator is None:
                value = float(constant) if prefer_proba else (1.0 if constant else -1.0)
                score = np.full(len(X), value)
            elif prefer_proba:
                positive = np.flatnonzero(estimator.classes_ == 1)[0]
                score = estimator.predict_proba(X)[:, positive]
            else:
                score = estimator.decision_function(X)
            columns.append(score)
        return np.column_stack(columns)


def optimize_thresholds(y_true, scores):
    if scores.shape != y_true.shape or not np.isfinite(scores).all():
        raise ValueError("Threshold tuning requires finite scores matching the labels")
    thresholds = np.empty(y_true.shape[1])
    for i, col in enumerate(y_true.T):
        s = scores[:, i]
        if not col.any():
            thresholds[i] = np.nextafter(float(s.max()), np.inf)
        elif col.all():
            thresholds[i] = float(s.min())
        else:
            precision, recall, candidates = precision_recall_curve(col, s)
            f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
            thresholds[i] = candidates[np.argmax(f1)]
    return thresholds


def aggregate_classification_metrics(report):
    """Expose the same thresholded metrics in detailed and tabular reports."""
    return {
        f"{metric}_{average}": float(report[f"{average} avg"][report_key])
        for average in ("micro", "macro", "weighted", "samples")
        for metric, report_key in (("precision", "precision"), ("recall", "recall"), ("f1", "f1-score"))
    }


def evaluate(y_true, scores, thresholds):
    """Evaluation never tunes thresholds, including when test labels are present."""
    if scores.shape != y_true.shape or not np.isfinite(scores).all():
        raise ValueError("Evaluation requires finite scores matching the labels")
    predictions = (scores >= thresholds).astype(int)
    metrics = {
        "hamming_loss": float(hamming_loss(y_true, predictions)),
        "exact_match_ratio": float(accuracy_score(y_true, predictions)),
    }
    report = classification_report(
        y_true, predictions, target_names=LABEL_COLS, zero_division=0, output_dict=True,
    )
    metrics.update(aggregate_classification_metrics(report))
    for average in ("micro", "macro", "weighted"):
        value = float(average_precision_score(y_true, scores, average=average))
        metrics[f"average_precision_{average}"] = value if np.isfinite(value) else None
    metrics["classification_report"] = report
    return metrics


def save_predictions(df, row_indices, scores, thresholds, path, *, has_labels):
    columns = (["id"] if "id" in df.columns else []) + [TEXT_COL]
    if has_labels:
        columns += LABEL_COLS
    out = df[columns].copy()
    out.insert(0, "source_row", row_indices)
    for i, label in enumerate(LABEL_COLS):
        out[f"pred_{label}"] = (scores[:, i] >= thresholds[i]).astype(int)
        out[f"score_{label}"] = scores[:, i]
    out.to_csv(path, index=False)


def write_json(path, value):
    """Atomic replacement prevents an interrupted write from damaging prior results."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)

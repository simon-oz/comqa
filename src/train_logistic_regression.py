# src/train_logistic_regression.py

import atexit
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    hamming_loss,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")


# ============================================================
# Configuration
# ============================================================

TEXT_COL = "question"

LABEL_COLS = [
    "retrieval",
    "comparison",
    "aggregation",
    "context_resolution",
    "temporal",
    "legal_citation",
    "evidence_extraction",
    "multi_document",
    "summary",
    "analyze",
    "evaluate",
    "recommend",
]

# The script will use the first training file that exists.
TRAIN_CANDIDATES = [
    "data/train_dataset.txt",
    "train_dataset.txt",
    "data/train_dataset.jsonl",
    "train_dataset.jsonl",
]

# External test dataset.
TEST_PATH = "data/test_dataset.jsonl"

# Embedding model.
# You can change this to a larger model if needed.
EMBEDDING_MODEL = "intfloat/e5-large-v2"

# Dev split size.
DEV_SIZE = 0.15
SEED = 42

ARTIFACT_DIR = Path("artifacts")
PREDICTION_DIR = Path("predictions")
LOG_DIR = Path("logs")

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Output logging
# ============================================================

_LOG_FILE = None
_LOG_PATH = None
_ORIGINAL_STDOUT = None
_ORIGINAL_STDERR = None
_TEE_STDOUT = None
_TEE_STDERR = None
_LOGGING_CLEANED_UP = False


class StreamTee:
    """
    Safely writes output to both the original stream and a log file.
    """

    @staticmethod
    def _to_text(message):
        if isinstance(message, bytes):
            return message.decode("utf-8", errors="ignore")
        return str(message)

    def __init__(self, original_stream, log_file):
        self.original_stream = original_stream
        self.log_file = log_file

    @property
    def closed(self):
        try:
            return bool(getattr(self.original_stream, "closed", False))
        except Exception:
            return False

    def write(self, message):
        try:
            text = self._to_text(message)
        except Exception:
            return

        # Write to terminal safely.
        try:
            if self.original_stream is not None and not getattr(self.original_stream, "closed", False):
                self.original_stream.write(text)
        except Exception:
            pass

        # Write to log file safely.
        try:
            if self.log_file is not None and not self.log_file.closed:
                self.log_file.write(text)
        except Exception:
            pass

    def flush(self):
        try:
            if self.original_stream is not None and not getattr(self.original_stream, "closed", False):
                self.original_stream.flush()
        except Exception:
            pass

        try:
            if self.log_file is not None and not self.log_file.closed:
                self.log_file.flush()
        except Exception:
            pass

    def isatty(self):
        try:
            return self.original_stream.isatty()
        except Exception:
            return False

    def fileno(self):
        if self.original_stream is None or getattr(self.original_stream, "closed", False):
            raise OSError("Underlying stream is closed.")
        return self.original_stream.fileno()

    def close(self):
        # Intentionally do nothing here.
        # Cleanup is handled by shutdown_output_logging().
        pass


def setup_output_logging():
    global _LOG_FILE
    global _LOG_PATH
    global _ORIGINAL_STDOUT
    global _ORIGINAL_STDERR
    global _TEE_STDOUT
    global _TEE_STDERR

    if _LOG_FILE is not None:
        return _LOG_PATH

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"train_logistic_regression_{timestamp}.log"

    _LOG_PATH = log_path
    _LOG_FILE = open(log_path, "a", encoding="utf-8")

    _ORIGINAL_STDOUT = sys.stdout
    _ORIGINAL_STDERR = sys.stderr

    _TEE_STDOUT = StreamTee(_ORIGINAL_STDOUT, _LOG_FILE)
    _TEE_STDERR = StreamTee(_ORIGINAL_STDERR, _LOG_FILE)

    sys.stdout = _TEE_STDOUT
    sys.stderr = _TEE_STDERR

    atexit.register(shutdown_output_logging)

    print(f"Logging output to: {log_path}")
    return log_path


def shutdown_output_logging():
    global _LOGGING_CLEANED_UP

    if _LOGGING_CLEANED_UP:
        return

    _LOGGING_CLEANED_UP = True

    # Restore original streams first.
    if _ORIGINAL_STDOUT is not None:
        sys.stdout = _ORIGINAL_STDOUT

    if _ORIGINAL_STDERR is not None:
        sys.stderr = _ORIGINAL_STDERR

    # Flush tee objects safely.
    for stream in (_TEE_STDOUT, _TEE_STDERR):
        try:
            if stream is not None:
                stream.flush()
        except Exception:
            pass

    # Flush and close the log file safely.
    if _LOG_FILE is not None:
        try:
            if not _LOG_FILE.closed:
                _LOG_FILE.flush()
                _LOG_FILE.close()
        except Exception:
            pass

    # Flush original stdout/stderr safely.
    for stream in (_ORIGINAL_STDOUT, _ORIGINAL_STDERR):
        try:
            if stream is not None and not getattr(stream, "closed", False):
                stream.flush()
        except Exception:
            pass


# ============================================================
# Data loading
# ============================================================

def find_train_path() -> str:
    for path in TRAIN_CANDIDATES:
        if Path(path).exists():
            return path

    raise FileNotFoundError(
        "Could not find training dataset. "
        "Please place it at data/train_dataset.txt, train_dataset.txt, "
        "data/train_dataset.jsonl, or train_dataset.jsonl."
    )


def load_jsonl(path: str):
    records = []

    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    df = pd.DataFrame(records)

    if TEXT_COL not in df.columns:
        raise ValueError(f"Dataset must contain text column: {TEXT_COL}")

    has_labels = all(col in df.columns for col in LABEL_COLS)

    # ========================================================
    # SURGICAL FIX:
    # Some labels, especially `retrieval`, contain values like 2.
    # For multi-label binary classification, convert any value > 0 to 1.
    # ========================================================
    for col in LABEL_COLS:
        if col not in df.columns:
            df[col] = 0

        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df[col] = (df[col] > 0).astype(int)

    df = df.dropna(subset=[TEXT_COL]).reset_index(drop=True)

    return df, has_labels


# ============================================================
# Train / dev split
# ============================================================

def split_train_dev(df: pd.DataFrame, dev_size: float = DEV_SIZE, seed: int = SEED):
    """
    Split training data into train and dev only.
    The external test_dataset.jsonl is used separately.
    """
    texts = df[TEXT_COL].astype(str).to_numpy()

    # SURGICAL FIX: ensure labels are binary before splitting.
    y = (df[LABEL_COLS].to_numpy() > 0).astype(int)

    # Try to stratify by exact multi-label combination.
    # If there are too many rare combinations, fall back to random split.
    combos = ["|".join(map(str, row)) for row in y]

    try:
        train_texts, dev_texts, y_train, y_dev = train_test_split(
            texts,
            y,
            test_size=dev_size,
            random_state=seed,
            stratify=combos,
        )
    except ValueError:
        print("Could not stratify by label combination. Using random train/dev split.")
        train_texts, dev_texts, y_train, y_dev = train_test_split(
            texts,
            y,
            test_size=dev_size,
            random_state=seed,
        )

    return (
        train_texts.tolist(),
        dev_texts.tolist(),
        np.asarray(y_train, dtype=int),
        np.asarray(y_dev, dtype=int),
    )


# ============================================================
# Safe one-vs-rest classifier
# ============================================================

class SafeOneVsRestClassifier:
    """
    A simple one-vs-rest wrapper that does not fail if a label has only
    one class in the training data.
    """

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimators_ = []
        self.constant_labels_ = []

        n_labels = y.shape[1]

        for label_idx in range(n_labels):
            col = y[:, label_idx]
            unique_values = np.unique(col)

            # If label is constant in train, store a constant predictor.
            if len(unique_values) < 2:
                self.estimators_.append(None)
                self.constant_labels_.append(int(unique_values[0]) if len(unique_values) > 0 else 0)
                continue

            est = clone(self.estimator)
            est.fit(X, col)

            self.estimators_.append(est)
            self.constant_labels_.append(None)

        return self

    def scores(self, X, prefer_proba: bool = True):
        n_samples = X.shape[0]
        score_columns = []

        for est, constant_label in zip(self.estimators_, self.constant_labels_):
            # Constant label from training.
            if est is None:
                if prefer_proba:
                    value = float(constant_label)
                else:
                    value = 1.0 if constant_label == 1 else -1.0

                score_columns.append(np.full(n_samples, value, dtype=float))
                continue

            s = None

            # Prefer probabilities if requested and available.
            if prefer_proba and hasattr(est, "predict_proba"):
                raw = np.asarray(est.predict_proba(X))

                if raw.ndim == 1:
                    s = raw
                elif raw.shape[1] == 1:
                    s = raw.ravel()
                else:
                    if hasattr(est, "classes_") and 1 in est.classes_:
                        pos_idx = int(np.where(est.classes_ == 1)[0][0])
                        s = raw[:, pos_idx]
                    else:
                        s = raw[:, -1]

            # Otherwise use decision_function if available.
            elif hasattr(est, "decision_function"):
                raw = np.asarray(est.decision_function(X))

                if raw.ndim == 1:
                    s = raw
                elif raw.shape[1] == 1:
                    s = raw.ravel()
                else:
                    if hasattr(est, "classes_") and 1 in est.classes_:
                        pos_idx = int(np.where(est.classes_ == 1)[0][0])
                        s = raw[:, pos_idx]
                    else:
                        s = raw[:, -1]

            # Fallback to probability if decision_function is unavailable.
            elif hasattr(est, "predict_proba"):
                raw = np.asarray(est.predict_proba(X))
                s = raw[:, -1]

            # Final fallback to hard predictions.
            else:
                s = est.predict(X).astype(float)

            score_columns.append(np.asarray(s, dtype=float).ravel())

        return np.column_stack(score_columns)


# ============================================================
# Embedding
# ============================================================

def embed_texts(train_texts, dev_texts, test_texts):
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Embedding model: {EMBEDDING_MODEL}")

    print(f"Encoding train texts: {len(train_texts)}")
    X_train = embedder.encode(
        train_texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print(f"Encoding dev texts: {len(dev_texts)}")
    X_dev = embedder.encode(
        dev_texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print(f"Encoding test texts: {len(test_texts)}")
    X_test = embedder.encode(
        test_texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_dev = scaler.transform(X_dev)
    X_test = scaler.transform(X_test)

    return (
        X_train.astype(np.float32),
        X_dev.astype(np.float32),
        X_test.astype(np.float32),
        scaler,
    )


# ============================================================
# Threshold tuning and evaluation
# ============================================================

def optimize_thresholds(y_true: np.ndarray, scores: np.ndarray) -> np.ndarray:
    # SURGICAL FIX: ensure labels are strictly binary.
    y_true = (np.asarray(y_true) > 0).astype(int)

    thresholds = np.zeros(y_true.shape[1], dtype=float)

    for label_idx in range(y_true.shape[1]):
        col = y_true[:, label_idx]
        s = np.asarray(scores[:, label_idx], dtype=float)
        s = np.nan_to_num(s, nan=-1e9)

        if len(s) == 0:
            thresholds[label_idx] = 0.5
            continue

        # No positives in dev: set threshold above max score.
        if col.sum() == 0:
            thresholds[label_idx] = np.max(s) + 1e-6
            continue

        # Only positives in dev: set threshold below min score.
        if col.sum() == len(col):
            thresholds[label_idx] = np.min(s) - 1e-6
            continue

        precision, recall, pr_thresholds = precision_recall_curve(col, s)

        if len(pr_thresholds) == 0:
            thresholds[label_idx] = 0.5
            continue

        f1 = (
            2 * precision[:-1] * recall[:-1]
            / (precision[:-1] + recall[:-1] + 1e-12)
        )

        if len(f1) == 0:
            thresholds[label_idx] = 0.5
            continue

        best_idx = int(np.nanargmax(f1))
        thresholds[label_idx] = float(pr_thresholds[best_idx])

    return thresholds


def safe_average_precision(y_true, y_score, average):
    try:
        return float(average_precision_score(y_true, y_score, average=average))
    except Exception:
        return float("nan")


def evaluate_predictions(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold=None,
    dataset_name: str = "",
):
    # SURGICAL FIX: ensure labels are strictly binary.
    y_true = (np.asarray(y_true) > 0).astype(int)

    if threshold is None:
        threshold = optimize_thresholds(y_true, scores)

    threshold = np.asarray(threshold)
    y_pred = (scores >= threshold).astype(int)

    print("=" * 90)
    print(f"EVALUATION: {dataset_name}")
    print("=" * 90)

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=LABEL_COLS,
            zero_division=0,
            digits=4,
        )
    )

    print(f"Hamming loss: {hamming_loss(y_true, y_pred):.4f}")
    print(f"Exact match ratio: {accuracy_score(y_true, y_pred):.4f}")

    print(f"F1 micro: {f1_score(y_true, y_pred, average='micro', zero_division=0):.4f}")
    print(f"F1 macro: {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"F1 weighted: {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1 samples: {f1_score(y_true, y_pred, average='samples', zero_division=0):.4f}")

    print(f"Average precision micro: {safe_average_precision(y_true, scores, 'micro'):.4f}")
    print(f"Average precision macro: {safe_average_precision(y_true, scores, 'macro'):.4f}")
    print(f"Average precision weighted: {safe_average_precision(y_true, scores, 'weighted'):.4f}")

    return y_pred, threshold


# ============================================================
# Prediction saving
# ============================================================

def save_predictions(
    df: pd.DataFrame,
    preds: np.ndarray,
    path: Path,
    include_true_labels: bool = True,
):
    out = df.copy()

    for i, col in enumerate(LABEL_COLS):
        out[f"pred_{col}"] = preds[:, i]

    keep_cols = []

    if "id" in out.columns:
        keep_cols.append("id")

    keep_cols.append(TEXT_COL)
    keep_cols.extend([f"pred_{col}" for col in LABEL_COLS])

    # Include true labels only if they were actually provided.
    if include_true_labels:
        keep_cols.extend([col for col in LABEL_COLS if col in out.columns])

    # Remove duplicates while preserving order.
    keep_cols = list(dict.fromkeys(keep_cols))

    out[keep_cols].to_csv(path, index=False)
    print(f"Saved predictions: {path}")


# ============================================================
# Main
# ============================================================

def main():
    train_path = find_train_path()

    print("=" * 90)
    print("Logistic Regression multi-label classifier")
    print("=" * 90)

    print(f"Loading training data from: {train_path}")
    train_df, train_has_labels = load_jsonl(train_path)

    if not train_has_labels:
        print("Warning: some label columns are missing in the training dataset.")

    # ========================================================
    # SURGICAL FIX:
    # Only split training data into train/dev.
    # External test_dataset.jsonl is loaded separately below.
    # ========================================================
    train_texts, dev_texts, y_train, y_dev = split_train_dev(train_df)

    print(f"Train size: {len(train_texts)}")
    print(f"Dev size: {len(dev_texts)}")

    print(f"Loading test data from: {TEST_PATH}")
    if not Path(TEST_PATH).exists():
        raise FileNotFoundError(f"Test dataset not found: {TEST_PATH}")

    test_df, test_has_labels = load_jsonl(TEST_PATH)
    test_texts = test_df[TEXT_COL].astype(str).tolist()

    # SURGICAL FIX:
    # Ensure test labels are binary too.
    y_test = (test_df[LABEL_COLS].to_numpy() > 0).astype(int)

    print(f"Test size: {len(test_texts)}")

    if not test_has_labels:
        print("Warning: test dataset does not contain all label columns.")
        print("Metrics will be skipped or may not be meaningful.")

    X_train, X_dev, X_test, scaler = embed_texts(
        train_texts=train_texts,
        dev_texts=dev_texts,
        test_texts=test_texts,
    )

    # ========================================================
    # TIMING: training
    # ========================================================
    print("\nTraining Logistic Regression...")
    train_start_time = time.perf_counter()

    model = SafeOneVsRestClassifier(
        LogisticRegression(
            C=0.3,
            class_weight="balanced",  # "balanced", None
            solver="liblinear",
            penalty="l2",
            max_iter=20000,
            random_state=SEED,
        )
    )

    model.fit(X_train, y_train)

    train_elapsed_time = time.perf_counter() - train_start_time

    print(f"Training time: {train_elapsed_time:.2f} seconds ({train_elapsed_time / 60:.2f} minutes)")

    # Tune thresholds on dev only.
    dev_scores = model.scores(X_dev, prefer_proba=True)
    _, thresholds = evaluate_predictions(
        y_dev,
        dev_scores,
        threshold=None,
        dataset_name="Logistic Regression dev",
    )

    # ========================================================
    # TIMING: test evaluation
    # ========================================================
    test_start_time = time.perf_counter()

    # Evaluate on external test set using dev-tuned thresholds.
    test_scores = model.scores(X_test, prefer_proba=True)

    if test_has_labels:
        test_preds, _ = evaluate_predictions(
            y_test,
            test_scores,
            threshold=thresholds,
            dataset_name="Logistic Regression test",
        )
    else:
        test_preds = (test_scores >= thresholds).astype(int)
        print("Test labels not found. Saving predictions without test metrics.")

    test_elapsed_time = time.perf_counter() - test_start_time

    # Save model artifacts.
    model_path = ARTIFACT_DIR / "logreg_ovr.joblib"
    threshold_path = ARTIFACT_DIR / "logreg_thresholds.npy"
    scaler_path = ARTIFACT_DIR / "logreg_scaler.joblib"

    dump(model, model_path)
    np.save(threshold_path, thresholds)
    dump(scaler, scaler_path)

    print(f"Saved model: {model_path}")
    print(f"Saved thresholds: {threshold_path}")
    print(f"Saved scaler: {scaler_path}")

    save_predictions(
        test_df,
        test_preds,
        PREDICTION_DIR / "test_logreg_predictions.csv",
        include_true_labels=test_has_labels,
    )

    # ========================================================
    # Save and print timing summary
    # ========================================================
    timing_info = {
        "train_seconds": float(train_elapsed_time),
        "train_minutes": float(train_elapsed_time / 60.0),
        "test_seconds": float(test_elapsed_time),
        "test_minutes": float(test_elapsed_time / 60.0),
    }

    timing_path = ARTIFACT_DIR / "logreg_timing.json"
    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(timing_info, f, indent=2)

    print("=" * 90)
    print("Timing summary")
    print("=" * 90)
    print(f"Training time: {train_elapsed_time:.2f} seconds ({train_elapsed_time / 60:.2f} minutes)")
    print(f"Test evaluation time: {test_elapsed_time:.2f} seconds ({test_elapsed_time / 60:.2f} minutes)")
    print(f"Saved timing: {timing_path}")


if __name__ == "__main__":
    setup_output_logging()

    try:
        main()
    finally:
        shutdown_output_logging()
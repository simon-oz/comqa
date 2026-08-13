# src/train_logreg_tfidf.py

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
from sklearn.feature_extraction.text import TfidfVectorizer
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

TRAIN_CANDIDATES = [
    "data/train_dataset.txt",
    "train_dataset.txt",
    "data/train_dataset.jsonl",
    "train_dataset.jsonl",
]

TEST_PATH = "data/test_dataset.jsonl"

DEV_SIZE = 0.15
SEED = 42

# Logistic Regression hyperparameters.
# Keep these the same as your current embedding-based Logistic Regression if desired.
LOGREG_C = 0.3
LOGREG_CLASS_WEIGHT = "balanced"
LOGREG_SOLVER = "liblinear"
LOGREG_PENALTY = "l2"
LOGREG_MAX_ITER = 20000

# TF-IDF configuration.
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 1
TFIDF_MAX_DF = 1.0
TFIDF_SUBLINEAR_TF = True
TFIDF_LOWERCASE = True
TFIDF_STRIP_ACCENTS = "unicode"

ARTIFACT_DIR = Path("artifacts")
PREDICTION_DIR = Path("predictions")
LOG_DIR = Path("logs")

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Safe output logging
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

        try:
            if self.original_stream is not None and not getattr(self.original_stream, "closed", False):
                self.original_stream.write(text)
        except Exception:
            pass

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
        try:
            if self.original_stream is None or getattr(self.original_stream, "closed", False):
                raise OSError("Underlying stream is closed.")
            return self.original_stream.fileno()
        except OSError:
            raise
        except Exception as exc:
            raise OSError("fileno is not available for this stream.") from exc

    def close(self):
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
    log_path = LOG_DIR / f"train_logreg_tfidf_{timestamp}.log"

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

    if _ORIGINAL_STDOUT is not None:
        sys.stdout = _ORIGINAL_STDOUT

    if _ORIGINAL_STDERR is not None:
        sys.stderr = _ORIGINAL_STDERR

    for stream in (_TEE_STDOUT, _TEE_STDERR):
        try:
            if stream is not None:
                stream.flush()
        except Exception:
            pass

    if _LOG_FILE is not None:
        try:
            if not _LOG_FILE.closed:
                _LOG_FILE.flush()
                _LOG_FILE.close()
        except Exception:
            pass

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
        "Please place it at data/train_dataset.txt or train_dataset.txt."
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

    # Binarize labels. Values like retrieval=2 become 1.
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
    y = (df[LABEL_COLS].to_numpy() > 0).astype(int)

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
            if est is None:
                if prefer_proba:
                    value = float(constant_label)
                else:
                    value = 1.0 if constant_label == 1 else -1.0

                score_columns.append(np.full(n_samples, value, dtype=float))
                continue

            s = None

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

            elif hasattr(est, "predict_proba"):
                raw = np.asarray(est.predict_proba(X))
                s = raw[:, -1]

            else:
                s = est.predict(X).astype(float)

            score_columns.append(np.asarray(s, dtype=float).ravel())

        return np.column_stack(score_columns)


# ============================================================
# TF-IDF feature extraction
# ============================================================

def build_tfidf_features(train_texts, dev_texts, test_texts):
    """
    Fit TF-IDF on train only, then transform train/dev/test.
    Returns sparse matrices compatible with LogisticRegression.
    """
    vectorizer = TfidfVectorizer(
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        sublinear_tf=TFIDF_SUBLINEAR_TF,
        lowercase=TFIDF_LOWERCASE,
        strip_accents=TFIDF_STRIP_ACCENTS,
        dtype=np.float32,
    )

    print("Fitting TF-IDF vectorizer on train texts...")
    X_train = vectorizer.fit_transform(train_texts)

    print("Transforming dev texts...")
    X_dev = vectorizer.transform(dev_texts)

    print("Transforming test texts...")
    X_test = vectorizer.transform(test_texts)

    print(f"TF-IDF feature shape: train={X_train.shape}, dev={X_dev.shape}, test={X_test.shape}")

    return X_train, X_dev, X_test, vectorizer


# ============================================================
# Threshold tuning and evaluation
# ============================================================

def safe_average_precision(y_true, y_score, average):
    try:
        return float(average_precision_score(y_true, y_score, average=average))
    except Exception:
        return float("nan")


def optimize_thresholds(y_true: np.ndarray, scores: np.ndarray) -> np.ndarray:
    y_true = (np.asarray(y_true) > 0).astype(int)
    scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=-1e9)

    thresholds = np.zeros(y_true.shape[1], dtype=float)

    for label_idx in range(y_true.shape[1]):
        col = y_true[:, label_idx]
        s = scores[:, label_idx]

        if len(s) == 0:
            thresholds[label_idx] = 0.5
            continue

        if col.sum() == 0:
            thresholds[label_idx] = np.max(s) + 1e-6
            continue

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


def evaluate_predictions(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold=None,
    dataset_name: str = "",
):
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

    if include_true_labels:
        keep_cols.extend([col for col in LABEL_COLS if col in out.columns])

    keep_cols = list(dict.fromkeys(keep_cols))

    out[keep_cols].to_csv(path, index=False)
    print(f"Saved predictions: {path}")


# ============================================================
# Main
# ============================================================

def main():
    total_start_time = time.perf_counter()
    timing = {}

    print("=" * 90)
    print("Logistic Regression multi-label classifier with TF-IDF features")
    print("=" * 90)

    train_path = find_train_path()

    print(f"Loading training data from: {train_path}")
    train_df, train_has_labels = load_jsonl(train_path)

    if not train_has_labels:
        print("Warning: some label columns are missing in the training dataset.")

    train_texts, dev_texts, y_train, y_dev = split_train_dev(train_df)

    print(f"Train size: {len(train_texts)}")
    print(f"Dev size: {len(dev_texts)}")

    print(f"Loading test data from: {TEST_PATH}")
    if not Path(TEST_PATH).exists():
        raise FileNotFoundError(f"Test dataset not found: {TEST_PATH}")

    test_df, test_has_labels = load_jsonl(TEST_PATH)
    test_texts = test_df[TEXT_COL].astype(str).tolist()
    y_test = (test_df[LABEL_COLS].to_numpy() > 0).astype(int)

    print(f"Test size: {len(test_texts)}")

    if not test_has_labels:
        print("Warning: test dataset does not contain all label columns.")
        print("Metrics will be skipped or may not be meaningful.")

    # --------------------------------------------------------
    # TF-IDF feature extraction
    # --------------------------------------------------------
    feature_start_time = time.perf_counter()

    X_train, X_dev, X_test, vectorizer = build_tfidf_features(
        train_texts=train_texts,
        dev_texts=dev_texts,
        test_texts=test_texts,
    )

    timing["tfidf_feature_extraction"] = float(time.perf_counter() - feature_start_time)

    # --------------------------------------------------------
    # Train Logistic Regression
    # --------------------------------------------------------
    train_start_time = time.perf_counter()

    print("\nTraining Logistic Regression on TF-IDF features...")
    model = SafeOneVsRestClassifier(
        LogisticRegression(
            C=LOGREG_C,
            class_weight=LOGREG_CLASS_WEIGHT,
            solver=LOGREG_SOLVER,
            penalty=LOGREG_PENALTY,
            max_iter=LOGREG_MAX_ITER,
            random_state=SEED,
        )
    )

    model.fit(X_train, y_train)

    timing["training"] = float(time.perf_counter() - train_start_time)

    # --------------------------------------------------------
    # Tune thresholds on dev only
    # --------------------------------------------------------
    dev_start_time = time.perf_counter()

    dev_scores = model.scores(X_dev, prefer_proba=True)
    _, thresholds = evaluate_predictions(
        y_dev,
        dev_scores,
        threshold=None,
        dataset_name="Logistic Regression TF-IDF dev",
    )

    timing["dev_evaluation_and_threshold_tuning"] = float(time.perf_counter() - dev_start_time)

    # --------------------------------------------------------
    # Evaluate on external test set using dev-tuned thresholds
    # --------------------------------------------------------
    test_start_time = time.perf_counter()

    test_scores = model.scores(X_test, prefer_proba=True)

    if test_has_labels:
        test_preds, _ = evaluate_predictions(
            y_test,
            test_scores,
            threshold=thresholds,
            dataset_name="Logistic Regression TF-IDF test",
        )
    else:
        test_preds = (test_scores >= thresholds).astype(int)
        print("Test labels not found. Saving predictions without test metrics.")

    timing["test_evaluation"] = float(time.perf_counter() - test_start_time)

    # --------------------------------------------------------
    # Save artifacts and predictions
    # --------------------------------------------------------
    save_start_time = time.perf_counter()

    model_path = ARTIFACT_DIR / "logreg_tfidf_ovr.joblib"
    threshold_path = ARTIFACT_DIR / "logreg_tfidf_thresholds.npy"
    vectorizer_path = ARTIFACT_DIR / "logreg_tfidf_vectorizer.joblib"

    dump(model, model_path)
    np.save(threshold_path, thresholds)
    dump(vectorizer, vectorizer_path)

    print(f"Saved model: {model_path}")
    print(f"Saved thresholds: {threshold_path}")
    print(f"Saved vectorizer: {vectorizer_path}")

    save_predictions(
        test_df,
        test_preds,
        PREDICTION_DIR / "test_logreg_tfidf_predictions.csv",
        include_true_labels=test_has_labels,
    )

    timing["saving_artifacts_and_predictions"] = float(time.perf_counter() - save_start_time)
    timing["total"] = float(time.perf_counter() - total_start_time)

    # --------------------------------------------------------
    # Save timing
    # --------------------------------------------------------
    timing_path = ARTIFACT_DIR / "logreg_tfidf_timing.json"
    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)

    print(f"Saved timing: {timing_path}")

    print("\n" + "=" * 90)
    print("Timing summary")
    print("=" * 90)

    for key, value in timing.items():
        print(f"{key:40s} {value:.2f} seconds ({value / 60:.2f} minutes)")


if __name__ == "__main__":
    setup_output_logging()

    try:
        main()
    finally:
        shutdown_output_logging()
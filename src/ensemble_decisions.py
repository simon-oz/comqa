# src/ensemble_decisions.py

import atexit
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    hamming_loss,
)


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

PRED_COLS = [f"pred_{col}" for col in LABEL_COLS]

# Prediction files produced by your individual model scripts.
MODEL_PREDICTION_FILES = {
    "logreg": "predictions/test_logreg_predictions.csv",
    "svm": "predictions/test_svm_predictions.csv",
    "deberta": "predictions/test_deberta_predictions.csv",
    "qwen": "predictions/test_qwen2_5_14b_predictions.csv",
}

# Model weights.
# Start with equal weights. Later you can increase weights for stronger models.
MODEL_WEIGHTS = {
    "logreg": 1.0,
    "svm": 1.0,
    "deberta": 1.0,
    "qwen": 1.0,
}

# Ensemble threshold.
#
# With equal weights and 4 models:
# - 0.5 means 2 or more votes predicts the label.
# - 0.5001 means strict majority, i.e. 3 or more votes with 4 models.
#
# If you want stricter majority voting, use 0.5001.
ENSEMBLE_THRESHOLD = 0.5

OUTPUT_PATH = Path("predictions/test_ensemble_predictions.csv")

LOG_DIR = Path("logs")
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
    log_path = LOG_DIR / f"ensemble_decisions_{timestamp}.log"

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
# Loading predictions
# ============================================================

def load_prediction_df(path: Path):
    if not path.exists():
        return None

    df = pd.read_csv(path)

    # Ensure prediction columns exist.
    for col in PRED_COLS:
        if col not in df.columns:
            df[col] = 0

        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
        df[col] = (df[col] > 0).astype(int)

    # Binarize true labels if they exist.
    for col in LABEL_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            df[col] = (df[col] > 0).astype(int)

    return df


def get_true_matrix(df: pd.DataFrame):
    if all(col in df.columns for col in LABEL_COLS):
        return (df[LABEL_COLS].to_numpy() > 0).astype(int)
    return None


# ============================================================
# Alignment
# ============================================================

def build_matrices(model_dfs):
    """
    Build aligned prediction matrices from multiple model prediction dataframes.
    """
    model_names_input = list(model_dfs.keys())

    if not model_names_input:
        raise ValueError("No model prediction files were loaded.")

    all_have_id = all("id" in model_dfs[name].columns for name in model_names_input)

    model_names = []
    pred_matrices = []
    true_matrix = None
    ids = None
    questions = None

    if all_have_id:
        # Use intersection of IDs across all models.
        common_ids = set(model_dfs[model_names_input[0]]["id"].tolist())

        for name in model_names_input[1:]:
            common_ids.intersection_update(set(model_dfs[name]["id"].tolist()))

        # Preserve row order from the first available prediction file.
        first_ids = model_dfs[model_names_input[0]]["id"].tolist()
        ordered_ids = [id_value for id_value in first_ids if id_value in common_ids]

        if not ordered_ids:
            raise ValueError("No common IDs found across prediction files.")

        ids = ordered_ids

        for name in model_names_input:
            df = model_dfs[name]

            df_aligned = (
                df.set_index("id")
                .loc[ordered_ids]
                .reset_index()
            )

            pred_matrices.append(df_aligned[PRED_COLS].to_numpy().astype(int))

            if true_matrix is None:
                true_matrix = get_true_matrix(df_aligned)

            if questions is None and TEXT_COL in df_aligned.columns:
                questions = df_aligned[TEXT_COL].tolist()

            model_names.append(name)

    else:
        # If no ID column exists, assume all files have the same row order.
        n_rows = len(model_dfs[model_names_input[0]])

        for name in model_names_input:
            df = model_dfs[name]

            if len(df) != n_rows:
                raise ValueError(
                    f"Prediction file for model '{name}' has {len(df)} rows, "
                    f"but expected {n_rows}. Cannot align without an ID column."
                )

            pred_matrices.append(df[PRED_COLS].to_numpy().astype(int))

            if true_matrix is None:
                true_matrix = get_true_matrix(df)

            if questions is None and TEXT_COL in df.columns:
                questions = df[TEXT_COL].tolist()

            model_names.append(name)

    return model_names, pred_matrices, true_matrix, ids, questions


# ============================================================
# Evaluation
# ============================================================

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, title: str):
    print("=" * 90)
    print(title)
    print("=" * 90)

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=LABEL_COLS,
            zero_division=0,
        )
    )

    print(f"Hamming loss: {hamming_loss(y_true, y_pred):.4f}")
    print(f"Exact match ratio: {accuracy_score(y_true, y_pred):.4f}")

    print(f"F1 micro: {f1_score(y_true, y_pred, average='micro', zero_division=0):.4f}")
    print(f"F1 macro: {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"F1 weighted: {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1 samples: {f1_score(y_true, y_pred, average='samples', zero_division=0):.4f}")


# ============================================================
# Ensemble
# ============================================================

def weighted_majority_vote(pred_matrices, model_names, weights_config):
    """
    pred_matrices: list of arrays, each shape (n_samples, n_labels)
    model_names: list of model names corresponding to pred_matrices
    weights_config: dictionary of model weights

    Returns:
        weighted_scores: shape (n_samples, n_labels), values in [0, 1]
        weights: normalized weights used
    """
    pred_array = np.stack(pred_matrices, axis=0).astype(float)

    weights = np.array(
        [float(weights_config.get(name, 1.0)) for name in model_names],
        dtype=float,
    )

    if weights.sum() <= 0:
        weights = np.ones_like(weights)

    weights = weights / weights.sum()

    # pred_array shape: (n_models, n_samples, n_labels)
    # weights shape: (n_models,)
    # weighted_scores shape: (n_samples, n_labels)
    weighted_scores = np.tensordot(weights, pred_array, axes=(0, 0))

    return weighted_scores, weights


def main():
    print("=" * 90)
    print("Multi-model decision ensemble")
    print("=" * 90)

    model_dfs = {}

    for model_name, path_str in MODEL_PREDICTION_FILES.items():
        path = Path(path_str)

        df = load_prediction_df(path)

        if df is None:
            print(f"Skipping {model_name}: prediction file not found at {path}")
            continue

        model_dfs[model_name] = df
        print(f"Loaded {model_name}: {path}")

    if not model_dfs:
        raise FileNotFoundError("No prediction files were found. Please run the individual models first.")

    model_names, pred_matrices, y_true, ids, questions = build_matrices(model_dfs)

    print("\nModels included in ensemble:")
    for name in model_names:
        print(f"  - {name}")

    # Evaluate individual models if true labels exist.
    if y_true is not None:
        print("\nEvaluating individual models...")
        for i, model_name in enumerate(model_names):
            evaluate_predictions(
                y_true,
                pred_matrices[i],
                title=f"Individual model: {model_name}",
            )

    # Weighted majority voting.
    weighted_scores, normalized_weights = weighted_majority_vote(
        pred_matrices=pred_matrices,
        model_names=model_names,
        weights_config=MODEL_WEIGHTS,
    )

    print("\nNormalized ensemble weights:")
    for name, weight in zip(model_names, normalized_weights):
        print(f"  {name:10s} {weight:.4f}")

    # Threshold can be scalar or one threshold per label.
    if np.isscalar(ENSEMBLE_THRESHOLD):
        thresholds = np.full(len(LABEL_COLS), float(ENSEMBLE_THRESHOLD), dtype=float)
    else:
        thresholds = np.asarray(ENSEMBLE_THRESHOLD, dtype=float)

        if len(thresholds) != len(LABEL_COLS):
            raise ValueError(
                f"ENSEMBLE_THRESHOLD must be a scalar or a list of length {len(LABEL_COLS)}."
            )

    ensemble_preds = (weighted_scores >= thresholds).astype(int)

    if y_true is not None:
        evaluate_predictions(
            y_true,
            ensemble_preds,
            title="Ensemble model",
        )

    # Save ensemble predictions.
    output_data = {}

    if ids is not None:
        output_data["id"] = ids

    if questions is not None:
        output_data[TEXT_COL] = questions

    for i, col in enumerate(LABEL_COLS):
        output_data[f"pred_{col}"] = ensemble_preds[:, i]
        output_data[f"ensemble_score_{col}"] = weighted_scores[:, i]

    if y_true is not None:
        for i, col in enumerate(LABEL_COLS):
            output_data[col] = y_true[:, i]

    output_df = pd.DataFrame(output_data)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved ensemble predictions: {OUTPUT_PATH}")


if __name__ == "__main__":
    setup_output_logging()

    try:
        main()
    finally:
        shutdown_output_logging()
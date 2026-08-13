# src/test_deberta_large.py
#
# Zero-shot test script for the RAW microsoft/deberta-v3-large base model.
# Uses ONLY:
#   - data/test_dataset.jsonl
#   - the raw microsoft/deberta-v3-large Masked Language Model (MLM)
#
# Does NOT use:
#   - training data
#   - dev data
#   - fine-tuned model checkpoints
#   - saved thresholds
#   - external NLI adapters (like MoritzLaurer)
#
# Method:
# Because the raw base model is an MLM, we use a Prompt-based Cloze approach.
# We score the probability of label "verbalizer" words at the [MASK] token.

import atexit
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    hamming_loss,
)
from transformers import AutoModelForMaskedLM, AutoTokenizer

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

LABEL_TO_IDX = {label: i for i, label in enumerate(LABEL_COLS)}

TEST_PATH = Path("data/test_dataset.jsonl")

# Strictly using the requested raw base model.
MODEL_NAME = "microsoft/deberta-v3-large"

# Verbalizer words for the [MASK] token.
LABEL_VERBALIZERS = {
    "retrieval": "retrieval",
    "comparison": "comparison",
    "aggregation": "aggregation",
    "context_resolution": "context",
    "temporal": "time",
    "legal_citation": "citation",
    "evidence_extraction": "extraction",
    "multi_document": "documents",
    "summary": "summary",
    "analyze": "analysis",
    "evaluate": "evaluation",
    "recommend": "recommendation",
}

# Because MLM probabilities are normalized across the 12 candidate labels,
# a threshold around 0.15 to 0.20 is usually a reasonable starting point.
ZERO_SHOT_THRESHOLD = 0.15

MAX_PROMPT_LENGTH = 510

# Optional: set to an integer for quick debugging, e.g. 5.
MAX_TEST_SAMPLES = None

OUTPUT_DIR = Path("deberta_large_zero_shot")
LOG_DIR = Path("logs")
PREDICTION_DIR = Path("predictions")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PROBS_PATH = OUTPUT_DIR / "test_deberta_mlm_zero_shot_probs.npy"
OUTPUT_RAW_PATH = OUTPUT_DIR / "test_deberta_mlm_zero_shot_raw_outputs.json"
OUTPUT_SUMMARY_PATH = OUTPUT_DIR / "test_deberta_mlm_zero_shot_summary.json"
OUTPUT_TIMING_PATH = OUTPUT_DIR / "test_deberta_mlm_zero_shot_timing.json"
OUTPUT_PREDICTIONS_PATH = PREDICTION_DIR / "test_deberta_mlm_zero_shot_predictions.csv"


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
    log_path = LOG_DIR / f"test_deberta_mlm_zero_shot_{timestamp}.log"

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
# JSON-safe scalar conversion
# ============================================================

def to_serializable_scalar(value):
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


# ============================================================
# Data loading
# ============================================================

def load_test_jsonl(path: Path):
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

    for col in LABEL_COLS:
        if col not in df.columns:
            df[col] = 0

        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df[col] = (df[col] > 0).astype(int)

    df = df.dropna(subset=[TEXT_COL]).reset_index(drop=True)

    if MAX_TEST_SAMPLES is not None:
        df = df.iloc[: int(MAX_TEST_SAMPLES)].reset_index(drop=True)

    return df, has_labels


# ============================================================
# MLM Zero-shot prediction
# ============================================================

def get_mlm_model():
    print(f"Loading raw base MLM model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    model.eval()
    return model, tokenizer


def predict_mlm_zero_shot(model, tokenizer, texts):
    mask_token = tokenizer.mask_token
    template = f"Question: {{}} The task is {mask_token}."

    print(f"Running MLM Cloze zero-shot inference on {len(texts)} examples...")

    # Pre-compute verbalizer token IDs
    verbalizer_ids = {}
    for label, word in LABEL_VERBALIZERS.items():
        ids = tokenizer.encode(word, add_special_tokens=False)
        if ids:
            verbalizer_ids[label] = ids[0]

    probs_matrix = np.zeros((len(texts), len(LABEL_COLS)), dtype=float)
    raw_outputs = []

    device = next(model.parameters()).device

    for i, text in enumerate(texts):
        prompt = template.format(text)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_PROMPT_LENGTH)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        mask_idx = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
        if len(mask_idx) == 0:
            # Fallback if mask got truncated
            mask_idx = torch.tensor([inputs["input_ids"].shape[1] - 2], device=device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, mask_idx[0], :]
            probs = torch.softmax(logits, dim=-1)

        label_scores = []
        for label in LABEL_COLS:
            vid = verbalizer_ids.get(label)
            if vid is not None:
                label_scores.append(probs[vid].item())
            else:
                label_scores.append(0.0)

        # Normalize across the 12 candidate labels so they sum to 1.0
        total_score = sum(label_scores)
        if total_score > 0:
            normalized_scores = [s / total_score for s in label_scores]
        else:
            normalized_scores = label_scores

        raw_item = {"labels": [], "scores": []}
        for j, label in enumerate(LABEL_COLS):
            probs_matrix[i, j] = normalized_scores[j]
            raw_item["labels"].append(label)
            raw_item["scores"].append(normalized_scores[j])

        raw_outputs.append(raw_item)

        if (i + 1) % 20 == 0 or (i + 1) == len(texts):
            print(f"Processed {i + 1}/{len(texts)} examples.")

    return probs_matrix, raw_outputs


# ============================================================
# Evaluation
# ============================================================

def safe_average_precision(y_true, y_score, average):
    try:
        return float(average_precision_score(y_true, y_score, average=average))
    except Exception:
        return float("nan")


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray):
    if len(y_true) == 0:
        print("No samples to evaluate.")
        return {}

    print("=" * 90)
    print("DeBERTa-v3-large (Raw MLM) zero-shot test report")
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

    metrics = {
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "exact_match_ratio": float(accuracy_score(y_true, y_pred)),
        "f1_micro": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_samples": float(f1_score(y_true, y_pred, average="samples", zero_division=0)),
        "ap_micro": safe_average_precision(y_true, probs, "micro"),
        "ap_macro": safe_average_precision(y_true, probs, "macro"),
        "ap_weighted": safe_average_precision(y_true, probs, "weighted"),
    }

    per_label_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    for i, label in enumerate(LABEL_COLS):
        metrics[f"f1_{label}"] = float(per_label_f1[i])

    print(f"Hamming loss: {metrics['hamming_loss']:.4f}")
    print(f"Exact match ratio: {metrics['exact_match_ratio']:.4f}")

    print(f"F1 micro: {metrics['f1_micro']:.4f}")
    print(f"F1 macro: {metrics['f1_macro']:.4f}")
    print(f"F1 weighted: {metrics['f1_weighted']:.4f}")
    print(f"F1 samples: {metrics['f1_samples']:.4f}")

    print(f"Average precision micro: {metrics['ap_micro']:.4f}")
    print(f"Average precision macro: {metrics['ap_macro']:.4f}")
    print(f"Average precision weighted: {metrics['ap_weighted']:.4f}")

    return metrics


# ============================================================
# Prediction saving
# ============================================================

def save_predictions(
    test_df: pd.DataFrame,
    preds: np.ndarray,
    probs: np.ndarray,
    include_true_labels: bool = True,
):
    out = test_df.copy()

    for i, label in enumerate(LABEL_COLS):
        out[f"pred_{label}"] = preds[:, i]
        out[f"prob_{label}"] = probs[:, i]

    keep_cols = []

    if "id" in out.columns:
        keep_cols.append("id")

    keep_cols.append(TEXT_COL)

    keep_cols.extend([f"pred_{label}" for label in LABEL_COLS])
    keep_cols.extend([f"prob_{label}" for label in LABEL_COLS])

    if include_true_labels:
        keep_cols.extend([label for label in LABEL_COLS if label in out.columns])

    keep_cols = list(dict.fromkeys(keep_cols))

    out[keep_cols].to_csv(OUTPUT_PREDICTIONS_PATH, index=False)
    print(f"Saved predictions: {OUTPUT_PREDICTIONS_PATH}")


def save_raw_outputs(raw_payload: dict):
    with open(OUTPUT_RAW_PATH, "w", encoding="utf-8") as f:
        json.dump(raw_payload, f, indent=2, ensure_ascii=False)

    print(f"Saved raw outputs: {OUTPUT_RAW_PATH}")


# ============================================================
# Main
# ============================================================

def main():
    total_start_time = time.perf_counter()

    print("=" * 90)
    print("DeBERTa-v3-large (Raw Base Model) zero-shot evaluation")
    print("=" * 90)

    print("Mode: MLM Cloze zero-shot classification")
    print("Train data used: no")
    print("Dev data used: no")
    print("Fine-tuned model used: no")
    print("Saved thresholds used: no")
    print("External NLI adapter used: no")

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test dataset not found: {TEST_PATH}")

    print(f"Loading test data from: {TEST_PATH}")
    test_df, test_has_labels = load_test_jsonl(TEST_PATH)

    test_texts = test_df[TEXT_COL].astype(str).tolist()
    y_true = (test_df[LABEL_COLS].to_numpy() > 0).astype(int)

    print(f"Test size: {len(test_df)}")

    if not test_has_labels:
        print("Warning: test dataset does not contain all label columns.")
        print("Metrics will be skipped.")

    model, tokenizer = get_mlm_model()

    inference_start_time = time.perf_counter()

    probs, raw_outputs = predict_mlm_zero_shot(model, tokenizer, test_texts)

    inference_elapsed_time = time.perf_counter() - inference_start_time

    preds = (probs >= ZERO_SHOT_THRESHOLD).astype(int)

    print("\n" + "=" * 90)
    print("Inference time summary")
    print("=" * 90)
    print(f"Inference time: {inference_elapsed_time:.2f} seconds ({inference_elapsed_time / 60:.2f} minutes)")

    print("\nZero-shot threshold (applied to normalized label probabilities):")
    print(f"  threshold: {ZERO_SHOT_THRESHOLD:.4f}")

    if test_has_labels:
        metrics = evaluate_predictions(y_true, preds, probs)
    else:
        print("Test labels not found. Skipping metrics.")
        metrics = {}

    # Save probabilities.
    np.save(str(OUTPUT_PROBS_PATH), probs.astype(np.float32))
    print(f"Saved probabilities: {OUTPUT_PROBS_PATH}")

    # Save predictions.
    save_predictions(
        test_df,
        preds,
        probs,
        include_true_labels=test_has_labels,
    )

    # Save raw zero-shot outputs.
    raw_payload = {
        "model_name": MODEL_NAME,
        "mode": "MLM Cloze zero-shot",
        "test_path": str(TEST_PATH),
        "test_size": int(len(test_df)),
        "max_test_samples": MAX_TEST_SAMPLES,
        "zero_shot_threshold": float(ZERO_SHOT_THRESHOLD),
        "outputs": [
            {
                "id": to_serializable_scalar(test_df.loc[i, "id"]) if "id" in test_df.columns else None,
                "question": str(test_df.loc[i, TEXT_COL]),
                "predicted_labels": [
                    LABEL_COLS[j] for j in range(len(LABEL_COLS)) if preds[i, j] == 1
                ],
                "zero_shot_labels": raw_outputs[i].get("labels", []) if i < len(raw_outputs) else [],
                "zero_shot_scores": raw_outputs[i].get("scores", []) if i < len(raw_outputs) else [],
            }
            for i in range(len(test_df))
        ],
    }

    save_raw_outputs(raw_payload)

    total_elapsed_time = time.perf_counter() - total_start_time

    timing_info = {
        "inference_seconds": float(inference_elapsed_time),
        "inference_minutes": float(inference_elapsed_time / 60.0),
        "total_seconds": float(total_elapsed_time),
        "total_minutes": float(total_elapsed_time / 60.0),
    }

    with open(OUTPUT_TIMING_PATH, "w", encoding="utf-8") as f:
        json.dump(timing_info, f, indent=2)

    summary = {
        "model_name": MODEL_NAME,
        "mode": "MLM Cloze zero-shot",
        "use_train_data": False,
        "use_dev_data": False,
        "use_fine_tuned_model": False,
        "use_saved_thresholds": False,
        "test_path": str(TEST_PATH),
        "test_size": int(len(test_df)),
        "zero_shot_threshold": float(ZERO_SHOT_THRESHOLD),
        "metrics": metrics,
        "timing": timing_info,
    }

    with open(OUTPUT_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 90)
    print("Final timing summary")
    print("=" * 90)
    print(f"Inference time: {inference_elapsed_time:.2f} seconds ({inference_elapsed_time / 60:.2f} minutes)")
    print(f"Total run time: {total_elapsed_time:.2f} seconds ({total_elapsed_time / 60:.2f} minutes)")

    print("\nSaved outputs:")
    print(f"Saved probabilities: {OUTPUT_PROBS_PATH}")
    print(f"Saved raw outputs: {OUTPUT_RAW_PATH}")
    print(f"Saved summary: {OUTPUT_SUMMARY_PATH}")
    print(f"Saved timing: {OUTPUT_TIMING_PATH}")
    print(f"Saved predictions: {OUTPUT_PREDICTIONS_PATH}")


if __name__ == "__main__":
    setup_output_logging()

    try:
        main()
    finally:
        shutdown_output_logging()
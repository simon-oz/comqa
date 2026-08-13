# src/test_qwen2_5_14b.py

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
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel

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

LABEL_TEXT = ", ".join(LABEL_COLS)

TEST_PATH = Path("data/test_dataset.jsonl")

MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
OUTPUT_DIR = Path("qwen2_5_14b_multilabel")
ADAPTER_DIR = OUTPUT_DIR / "final"
THRESHOLDS_PATH = OUTPUT_DIR / "dev_thresholds.npy"

PREDICTION_DIR = Path("predictions")
LOG_DIR = Path("logs")

PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PROBS_PATH = OUTPUT_DIR / "test_qwen2_5_14b_probs.npy"
OUTPUT_SUMMARY_PATH = OUTPUT_DIR / "test_qwen2_5_14b_evaluation_summary.json"
OUTPUT_TIMING_PATH = OUTPUT_DIR / "test_qwen2_5_14b_timing.json"
OUTPUT_PREDICTIONS_PATH = PREDICTION_DIR / "test_qwen2_5_14b_eval_predictions.csv"

# If True, use saved dev-tuned thresholds.
# If False, use 0.5 for all labels.
USE_DEV_THRESHOLDS = True

# Inference settings.
# Qwen2.5-14B is large, so keep batch size small.
BATCH_SIZE = 2
MAX_LENGTH = 512

# Set True if you want BF16 inference.
# On H100 this is usually the best balance of speed and memory.
USE_BF16_INFERENCE = True

# Set True only if the Qwen model was trained with QLoRA 4-bit.
# Your current training script default was USE_QLORA=False.
USE_QLORA = False


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
    log_path = LOG_DIR / f"test_qwen2_5_14b_{timestamp}.log"

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
# Helpers
# ============================================================

def cuda_bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False

    if not hasattr(torch.cuda, "is_bf16_supported"):
        return False

    try:
        return bool(torch.cuda.is_bf16_supported())
    except Exception:
        return False


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30, 30)
    return 1.0 / (1.0 + np.exp(-x))


def safe_average_precision(y_true, y_score, average):
    try:
        return float(average_precision_score(y_true, y_score, average=average))
    except Exception:
        return float("nan")


def build_text(question: str) -> str:
    """
    Must match the prompt used during Qwen training.
    """
    return (
        "Classify the required task labels for the following tax/legal question. "
        f"Available labels: {LABEL_TEXT}.\n"
        f"Question: {question}"
    )


def load_jsonl(path: Path):
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


def load_thresholds(path: Path):
    if USE_DEV_THRESHOLDS and path.exists():
        thresholds = np.load(path)

        if thresholds.shape == (len(LABEL_COLS),):
            print(f"Loaded dev-tuned thresholds from: {path}")
            return thresholds.astype(float)

        print(
            f"Warning: threshold file {path} has unexpected shape {thresholds.shape}. "
            f"Falling back to 0.5 for all labels."
        )

    elif USE_DEV_THRESHOLDS:
        print(f"Warning: threshold file not found at {path}. Using 0.5 for all labels.")

    return np.full(len(LABEL_COLS), 0.5, dtype=float)


def load_qwen_model():
    print(f"Loading tokenizer from: {ADAPTER_DIR}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            ADAPTER_DIR,
            trust_remote_code=True,
        )
    except Exception:
        print(f"Tokenizer not found in {ADAPTER_DIR}. Falling back to base model tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Decoder-only models should use left padding for classification.
    tokenizer.padding_side = "left"

    if USE_BF16_INFERENCE and cuda_bf16_supported():
        torch_dtype = torch.bfloat16
        print("Using BF16 inference.")
    else:
        torch_dtype = torch.float32
        print("Using FP32 inference.")

    bnb_config = None

    if USE_QLORA:
        print("Using QLoRA 4-bit quantization for inference.")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if cuda_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model_kwargs = dict(
        num_labels=len(LABEL_COLS),
        problem_type="multi_label_classification",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto",
    )

    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config

    print(f"Loading base model: {MODEL_NAME}")

    try:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            dtype=torch_dtype,
            **model_kwargs,
        )
    except TypeError:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch_dtype,
            **model_kwargs,
        )

    if base_model.config.pad_token_id is None:
        base_model.config.pad_token_id = tokenizer.pad_token_id

    print(f"Loading LoRA adapter from: {ADAPTER_DIR}")

    model = PeftModel.from_pretrained(
        base_model,
        ADAPTER_DIR,
    )

    model.eval()

    return model, tokenizer


def predict_probs(model, tokenizer, texts):
    device = next(model.parameters()).device

    all_logits = []

    for start in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[start:start + BATCH_SIZE]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits.float().cpu().numpy()
        all_logits.append(logits)

    logits = np.vstack(all_logits)
    probs = sigmoid(logits)

    return probs, logits


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray, title: str):
    print("=" * 90)
    print(title)
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

    print(f"Average precision micro: {safe_average_precision(y_true, probs, 'micro'):.4f}")
    print(f"Average precision macro: {safe_average_precision(y_true, probs, 'macro'):.4f}")
    print(f"Average precision weighted: {safe_average_precision(y_true, probs, 'weighted'):.4f}")


def build_metrics(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray):
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
    for label_name, value in zip(LABEL_COLS, per_label_f1):
        metrics[f"f1_{label_name}"] = float(value)

    return metrics


def save_predictions(
    df: pd.DataFrame,
    preds: np.ndarray,
    probs: np.ndarray,
    path: Path,
    include_true_labels: bool = True,
):
    out = df.copy()

    for i, col in enumerate(LABEL_COLS):
        out[f"pred_{col}"] = preds[:, i]
        out[f"prob_{col}"] = probs[:, i]

    keep_cols = []

    if "id" in out.columns:
        keep_cols.append("id")

    keep_cols.append(TEXT_COL)

    keep_cols.extend([f"pred_{col}" for col in LABEL_COLS])
    keep_cols.extend([f"prob_{col}" for col in LABEL_COLS])

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

    print("=" * 90)
    print("Qwen2.5-14B-Instruct direct test evaluation")
    print("=" * 90)

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"BF16 supported: {cuda_bf16_supported()}")
    else:
        print("Warning: Qwen2.5-14B is extremely large and may not run on CPU.")

    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test dataset not found: {TEST_PATH}")

    if not ADAPTER_DIR.exists():
        raise FileNotFoundError(
            f"Qwen adapter directory not found: {ADAPTER_DIR}. "
            "Please train the model first using train_qwen2_5_14b.py."
        )

    print(f"Loading test data from: {TEST_PATH}")
    test_df, test_has_labels = load_jsonl(TEST_PATH)

    test_texts_raw = test_df[TEXT_COL].astype(str).tolist()
    test_texts = [build_text(question) for question in test_texts_raw]

    y_test = (test_df[LABEL_COLS].to_numpy() > 0).astype(int)

    print(f"Test size: {len(test_texts)}")

    if not test_has_labels:
        print("Warning: test dataset does not contain all label columns.")
        print("Metrics will be skipped or may not be meaningful.")

    thresholds = load_thresholds(THRESHOLDS_PATH)

    print("\nThresholds:")
    for label_name, threshold in zip(LABEL_COLS, thresholds):
        print(f"  {label_name:25s} {threshold:.4f}")

    model, tokenizer = load_qwen_model()

    print("\nRunning inference on test set...")
    inference_start_time = time.perf_counter()

    probs, logits = predict_probs(model, tokenizer, test_texts)

    inference_elapsed_time = time.perf_counter() - inference_start_time

    print(f"Inference time: {inference_elapsed_time:.2f} seconds ({inference_elapsed_time / 60:.2f} minutes)")

    preds = (probs >= thresholds).astype(int)

    if test_has_labels:
        evaluate_predictions(
            y_test,
            preds,
            probs,
            title="Qwen2.5-14B-Instruct test report",
        )

        metrics = build_metrics(y_test, preds, probs)
    else:
        print("Test labels not found. Skipping metrics.")
        metrics = {}

    # Save probabilities.
    np.save(str(OUTPUT_PROBS_PATH), probs.astype(np.float32))
    print(f"Saved probabilities: {OUTPUT_PROBS_PATH}")

    # Save evaluation summary.
    summary = {
        "model_name": MODEL_NAME,
        "adapter_dir": str(ADAPTER_DIR),
        "test_path": str(TEST_PATH),
        "test_size": int(len(test_texts)),
        "use_dev_thresholds": bool(USE_DEV_THRESHOLDS),
        "thresholds": {label: float(threshold) for label, threshold in zip(LABEL_COLS, thresholds)},
        "metrics": metrics,
        "inference_seconds": float(inference_elapsed_time),
        "inference_minutes": float(inference_elapsed_time / 60.0),
    }

    with open(OUTPUT_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved evaluation summary: {OUTPUT_SUMMARY_PATH}")

    # Save predictions.
    save_predictions(
        test_df,
        preds,
        probs,
        OUTPUT_PREDICTIONS_PATH,
        include_true_labels=test_has_labels,
    )

    total_elapsed_time = time.perf_counter() - total_start_time

    timing_info = {
        "inference_seconds": float(inference_elapsed_time),
        "inference_minutes": float(inference_elapsed_time / 60.0),
        "total_seconds": float(total_elapsed_time),
        "total_minutes": float(total_elapsed_time / 60.0),
    }

    with open(OUTPUT_TIMING_PATH, "w", encoding="utf-8") as f:
        json.dump(timing_info, f, indent=2)

    print("\n" + "=" * 90)
    print("Final timing summary")
    print("=" * 90)
    print(f"Inference time: {inference_elapsed_time:.2f} seconds ({inference_elapsed_time / 60:.2f} minutes)")
    print(f"Total run time: {total_elapsed_time:.2f} seconds ({total_elapsed_time / 60:.2f} minutes)")
    print(f"Saved timing: {OUTPUT_TIMING_PATH}")


if __name__ == "__main__":
    setup_output_logging()

    try:
        main()
    finally:
        shutdown_output_logging()
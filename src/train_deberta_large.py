# src/train_deberta_large.py

import atexit
import inspect
import json
import sys
import time
import warnings
from dataclasses import fields
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

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

# DeBERTa model.
# You can also use:
# MODEL_NAME = "microsoft/deberta-large"
MODEL_NAME = "microsoft/deberta-v3-large"

OUTPUT_DIR = Path("deberta_large_multilabel")
LOG_DIR = Path("logs")
PREDICTION_DIR = Path("predictions")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

MAX_LENGTH = 512

# Reduce batch size if you get CUDA out-of-memory errors.
TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 1

LEARNING_RATE = 7e-6
NUM_EPOCHS = 12

# Limits extreme positive weights for rare labels.
POS_WEIGHT_CLIP = 5.0

DEV_SIZE = 0.20
SEED = 42


# ============================================================
# Robust helpers
# ============================================================

def cuda_bf16_supported() -> bool:
    """
    Returns True only if CUDA is available and BF16 is supported.
    """
    if not torch.cuda.is_available():
        return False

    if not hasattr(torch.cuda, "is_bf16_supported"):
        return False

    try:
        return bool(torch.cuda.is_bf16_supported())
    except Exception:
        return False


def get_training_argument_names():
    """
    Get valid TrainingArguments field names for the installed Transformers version.

    This helps support both older and newer versions where arguments may be
    renamed or removed.
    """
    try:
        return {f.name for f in fields(TrainingArguments)}
    except Exception:
        try:
            return set(inspect.signature(TrainingArguments.__init__).parameters.keys())
        except Exception:
            return set()


def get_eval_strategy_argument_name(training_arg_names):
    """
    Newer Transformers versions use eval_strategy.
    Older versions used evaluation_strategy.
    """
    if "eval_strategy" in training_arg_names:
        return "eval_strategy"

    if "evaluation_strategy" in training_arg_names:
        return "evaluation_strategy"

    raise RuntimeError(
        "Could not find a valid evaluation strategy argument in TrainingArguments. "
        "Please check your Transformers version."
    )


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
    log_path = LOG_DIR / f"train_deberta_large_{timestamp}.log"

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
    # This is important for DeBERTa because BCEWithLogitsLoss should
    # use binary targets.
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
# Metrics and thresholding
# ============================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -30, 30)
    return 1.0 / (1.0 + np.exp(-x))


def safe_average_precision(y_true, y_score, average):
    try:
        return float(average_precision_score(y_true, y_score, average=average))
    except Exception:
        return float("nan")


def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # SURGICAL FIX: ensure labels are strictly binary.
    labels = (np.asarray(labels) > 0).astype(int)

    probs = sigmoid(logits)
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "f1_micro": float(f1_score(labels, preds, average="micro", zero_division=0)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "f1_samples": float(f1_score(labels, preds, average="samples", zero_division=0)),
        "ap_micro": safe_average_precision(labels, probs, "micro"),
        "ap_macro": safe_average_precision(labels, probs, "macro"),
        "ap_weighted": safe_average_precision(labels, probs, "weighted"),
    }

    per_label_f1 = f1_score(labels, preds, average=None, zero_division=0)
    for label_name, value in zip(LABEL_COLS, per_label_f1):
        metrics[f"f1_{label_name}"] = float(value)

    return metrics


def optimize_thresholds(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    # SURGICAL FIX: ensure labels are strictly binary.
    labels = (np.asarray(labels) > 0).astype(int)

    probs = sigmoid(logits)
    thresholds = np.zeros(labels.shape[1], dtype=float)

    for label_idx in range(labels.shape[1]):
        col = labels[:, label_idx]
        s = probs[:, label_idx]

        if len(s) == 0:
            thresholds[label_idx] = 0.5
            continue

        # No positives in dev: set threshold above max probability.
        if col.sum() == 0:
            thresholds[label_idx] = 1.01
            continue

        # Only positives in dev: set threshold below min probability.
        if col.sum() == len(col):
            thresholds[label_idx] = 0.0
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


def report_with_threshold(
    labels: np.ndarray,
    logits: np.ndarray,
    thresholds: np.ndarray,
    title: str,
):
    # SURGICAL FIX: ensure labels are strictly binary.
    labels = (np.asarray(labels) > 0).astype(int)

    probs = sigmoid(logits)
    preds = (probs >= thresholds).astype(int)

    print("=" * 90)
    print(title)
    print("=" * 90)

    print(
        classification_report(
            labels,
            preds,
            target_names=LABEL_COLS,
            zero_division=0,
            digits=4,
        )
    )

    print(f"F1 micro: {f1_score(labels, preds, average='micro', zero_division=0):.4f}")
    print(f"F1 macro: {f1_score(labels, preds, average='macro', zero_division=0):.4f}")
    print(f"F1 weighted: {f1_score(labels, preds, average='weighted', zero_division=0):.4f}")
    print(f"F1 samples: {f1_score(labels, preds, average='samples', zero_division=0):.4f}")

    print(f"Average precision micro: {safe_average_precision(labels, probs, 'micro'):.4f}")
    print(f"Average precision macro: {safe_average_precision(labels, probs, 'macro'):.4f}")
    print(f"Average precision weighted: {safe_average_precision(labels, probs, 'weighted'):.4f}")


# ============================================================
# Positive weighting for BCE loss
# ============================================================

def make_pos_weight(y: np.ndarray, clip: float = POS_WEIGHT_CLIP) -> torch.Tensor:
    """
    Positive class weight for BCEWithLogitsLoss.

    For label j:
        pos_weight[j] = number_negative / number_positive

    The value is clipped to avoid extremely large weights for rare labels.
    """
    # SURGICAL FIX: ensure labels are binary.
    y = (np.asarray(y) > 0).astype(int)

    pos_counts = y.sum(axis=0).astype(np.float32)
    neg_counts = len(y) - pos_counts

    weights = np.ones_like(pos_counts)

    positive_mask = pos_counts > 0
    weights[positive_mask] = neg_counts[positive_mask] / pos_counts[positive_mask]

    weights = np.clip(weights, 1.0, clip)

    return torch.tensor(weights, dtype=torch.float)


# ============================================================
# Custom data collator and trainer
# ============================================================

class MultiLabelDataCollator(DataCollatorWithPadding):
    """
    Data collator that keeps multi-label targets as float tensors.
    BCEWithLogitsLoss expects float labels, not long/int labels.
    """

    def __call__(self, features):
        labels = [feature.pop("labels", None) for feature in features]
        batch = super().__call__(features)

        if labels and labels[0] is not None:
            # SURGICAL FIX: convert to float and binarize defensively.
            labels_tensor = torch.tensor(labels, dtype=torch.float)
            labels_tensor = (labels_tensor > 0).float()
            batch["labels"] = labels_tensor

        return batch


class WeightedBCETrainer(Trainer):
    """
    Trainer that uses BCEWithLogitsLoss with optional pos_weight.
    """

    def __init__(self, pos_weight=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels", None)

        inputs_without_labels = {
            key: value for key, value in inputs.items() if key != "labels"
        }

        outputs = model(**inputs_without_labels)
        logits = outputs.logits

        if labels is None:
            loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=logits.device)
        else:
            # SURGICAL FIX: defensively ensure targets are binary.
            labels = (labels > 0).float()

            pos_weight = None
            if self.pos_weight is not None:
                pos_weight = self.pos_weight.to(logits.device)

            loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss = loss_fct(logits, labels.float())

        return (loss, outputs) if return_outputs else loss


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
    print("DeBERTa-large multi-label classifier")
    print("=" * 90)

    print(f"CUDA available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("Warning: DeBERTa-large on CPU will be very slow.")

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

    # SURGICAL FIX: ensure test labels are binary too.
    y_test = (test_df[LABEL_COLS].to_numpy() > 0).astype(int)

    print(f"Test size: {len(test_texts)}")

    if not test_has_labels:
        print("Warning: test dataset does not contain all label columns.")
        print("Metrics will be skipped or may not be meaningful.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

    train_ds = Dataset.from_dict(
        {
            "text": train_texts,
            "labels": y_train.tolist(),
        }
    )

    dev_ds = Dataset.from_dict(
        {
            "text": dev_texts,
            "labels": y_dev.tolist(),
        }
    )

    test_ds = Dataset.from_dict(
        {
            "text": test_texts,
            "labels": y_test.tolist(),
        }
    )

    print("Tokenizing train dataset...")
    train_ds = train_ds.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing train",
    )

    print("Tokenizing dev dataset...")
    dev_ds = dev_ds.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing dev",
    )

    print("Tokenizing test dataset...")
    test_ds = test_ds.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing test",
    )

    # ========================================================
    # SURGICAL FIX:
    # Load model in FP32 to avoid AMP FP16 gradient-unscaling issues.
    # ========================================================
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_COLS),
        problem_type="multi_label_classification",
        torch_dtype=torch.float32,
    )

    model.config.id2label = {i: label for i, label in enumerate(LABEL_COLS)}
    model.config.label2id = {label: i for i, label in enumerate(LABEL_COLS)}

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model.config.use_cache = False

    # ========================================================
    # Robust TrainingArguments construction.
    #
    # Fixes:
    # 1. eval_strategy vs evaluation_strategy
    # 2. FP16/BF16 precision handling
    # ========================================================
    training_arg_names = get_training_argument_names()
    eval_strategy_key = get_eval_strategy_argument_name(training_arg_names)

    use_bf16 = cuda_bf16_supported()

    print(f"Using eval argument: {eval_strategy_key}")
    print(f"BF16 supported: {use_bf16}")
    print("FP16 disabled to avoid AMP gradient-unscaling issues.")

    training_kwargs = dict(
        output_dir=str(OUTPUT_DIR),
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.05,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        save_strategy="epoch",
        logging_steps=25,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="ap_macro",
        greater_is_better=True,
        report_to="none",
        seed=SEED,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )

    # Add evaluation strategy using the correct argument name.
    training_kwargs[eval_strategy_key] = "epoch"

    # ========================================================
    # Precision handling:
    # - Disable FP16.
    # - Use BF16 only if supported.
    # - Otherwise fall back to FP32.
    # ========================================================
    if "fp16" in training_arg_names:
        training_kwargs["fp16"] = False

    if "bf16" in training_arg_names:
        training_kwargs["bf16"] = use_bf16

    # If the installed Transformers version has removed/renamed some arguments,
    # filter to only valid arguments.
    if training_arg_names:
        training_kwargs = {
            key: value
            for key, value in training_kwargs.items()
            if key in training_arg_names
        }

        # Re-add required eval strategy after filtering.
        training_kwargs[eval_strategy_key] = "epoch"

        # Re-add precision settings after filtering.
        if "fp16" in training_arg_names:
            training_kwargs["fp16"] = False

        if "bf16" in training_arg_names:
            training_kwargs["bf16"] = use_bf16

    training_args = TrainingArguments(**training_kwargs)

    data_collator = MultiLabelDataCollator(tokenizer=tokenizer)

    pos_weight = make_pos_weight(y_train)

    trainer_common_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
        ],
    )

    # Newer Transformers versions prefer `processing_class`.
    # Older versions use `tokenizer`.
    try:
        trainer = WeightedBCETrainer(
            pos_weight=pos_weight,
            processing_class=tokenizer,
            **trainer_common_kwargs,
        )
    except TypeError:
        trainer = WeightedBCETrainer(
            pos_weight=pos_weight,
            tokenizer=tokenizer,
            **trainer_common_kwargs,
        )

    # ========================================================
    # TIMING: training
    # ========================================================
    print("\nStarting DeBERTa training...")
    train_start_time = time.perf_counter()

    train_output = trainer.train()

    train_elapsed_time = time.perf_counter() - train_start_time

    print(f"Training time: {train_elapsed_time:.2f} seconds ({train_elapsed_time / 60:.2f} minutes)")

    if train_output is not None and hasattr(train_output, "metrics") and "train_runtime" in train_output.metrics:
        hf_train_runtime = float(train_output.metrics["train_runtime"])
        print(f"Hugging Face train_runtime: {hf_train_runtime:.2f} seconds ({hf_train_runtime / 60:.2f} minutes)")

    print("\nTuning thresholds on dev set...")
    dev_output = trainer.predict(dev_ds)
    dev_logits = np.asarray(dev_output.predictions)
    dev_labels = np.asarray(dev_output.label_ids)

    thresholds = optimize_thresholds(dev_logits, dev_labels)

    report_with_threshold(
        dev_labels,
        dev_logits,
        thresholds,
        title="DeBERTa dev report using thresholds tuned on dev",
    )

    # ========================================================
    # TIMING: test evaluation
    # ========================================================
    print("\nEvaluating external test set using dev-tuned thresholds...")
    test_start_time = time.perf_counter()

    test_output = trainer.predict(test_ds)
    test_logits = np.asarray(test_output.predictions)
    test_labels = np.asarray(test_output.label_ids)

    test_probs = sigmoid(test_logits)
    test_preds = (test_probs >= thresholds).astype(int)

    if test_has_labels:
        report_with_threshold(
            test_labels,
            test_logits,
            thresholds,
            title="DeBERTa test report using dev-tuned thresholds",
        )
    else:
        print("Test labels not found. Saving predictions without test metrics.")

    test_elapsed_time = time.perf_counter() - test_start_time

    # ========================================================
    # Save model, tokenizer, thresholds, predictions
    # ========================================================
    final_dir = OUTPUT_DIR / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    np.save(str(OUTPUT_DIR / "dev_thresholds.npy"), thresholds)

    print(f"Saved model: {final_dir}")
    print(f"Saved tokenizer: {final_dir}")
    print(f"Saved thresholds: {OUTPUT_DIR / 'dev_thresholds.npy'}")

    save_predictions(
        test_df,
        test_preds,
        PREDICTION_DIR / "test_deberta_predictions.csv",
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

    if train_output is not None and hasattr(train_output, "metrics") and "train_runtime" in train_output.metrics:
        timing_info["hf_train_runtime_seconds"] = float(train_output.metrics["train_runtime"])
        timing_info["hf_train_runtime_minutes"] = float(train_output.metrics["train_runtime"]) / 60.0

    timing_path = OUTPUT_DIR / "timing.json"
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
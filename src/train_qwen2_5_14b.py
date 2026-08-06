# train_qwen2_5_14b.py

import atexit
import inspect
import json
import sys
import warnings
from dataclasses import fields
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    hamming_loss,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)

try:
    from skmultilearn.model_selection import iterative_train_test_split
except Exception:
    iterative_train_test_split = None

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

MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"

OUTPUT_DIR = Path("qwen2_5_14b_multilabel")
LOG_DIR = Path("logs")
PREDICTION_DIR = Path("predictions")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Main training configuration
# ------------------------------------------------------------

# If False: BF16 LoRA. Usually better quality if memory allows.
# If True: 4-bit QLoRA. Use this if you get CUDA OOM.
USE_QLORA = False

# LoRA capacity.
# If underfitting, increase LORA_R and LORA_ALPHA, e.g. 64/128.
# If overfitting, reduce to 16/32.
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05

# Learning rate for LoRA + classification head.
# If overfitting, try 5e-5 or 3e-5.
# If underfitting, try 2e-4.
LEARNING_RATE = 1e-4

NUM_EPOCHS = 8

TRAIN_BATCH_SIZE = 2
EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8

# Positive weighting for rare labels.
# 1.0 approximately disables upweighting.
# Try 1.0, 3.0, 5.0.
POS_WEIGHT_CLIP = 3.0

MAX_LENGTH = 512

DEV_SIZE = 0.20
SEED = 42

LABEL_TEXT = ", ".join(LABEL_COLS)


# ============================================================
# Robust helpers
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


def get_training_argument_names():
    try:
        return {f.name for f in fields(TrainingArguments)}
    except Exception:
        try:
            return set(inspect.signature(TrainingArguments.__init__).parameters.keys())
        except Exception:
            return set()


def get_eval_strategy_argument_name(training_arg_names):
    if not training_arg_names:
        return "eval_strategy"

    if "eval_strategy" in training_arg_names:
        return "eval_strategy"

    if "evaluation_strategy" in training_arg_names:
        return "evaluation_strategy"

    return "eval_strategy"


# ============================================================
# Output logging
# ============================================================

class StreamTee:
    @staticmethod
    def _to_text(message):
        if isinstance(message, bytes):
            return message.decode("utf-8", errors="ignore")
        return str(message)

    def __init__(self, original_stream, log_file):
        self.original_stream = original_stream
        self.log_file = log_file

    def write(self, message):
        text = self._to_text(message)
        self.original_stream.write(text)
        self.log_file.write(text)

    def flush(self):
        self.original_stream.flush()
        self.log_file.flush()

    def isatty(self):
        return self.original_stream.isatty()

    def fileno(self):
        return self.original_stream.fileno()


def setup_output_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"train_qwen2_5_14b_{timestamp}.log"

    log_file = open(log_path, "a", encoding="utf-8")

    sys.stdout = StreamTee(sys.stdout, log_file)
    sys.stderr = StreamTee(sys.stderr, log_file)

    atexit.register(log_file.close)

    print(f"Logging output to: {log_path}")
    return log_path


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

def _to_dense_labels(y, n_labels=None):
    """
    Convert label output from iterative_train_test_split to dense numpy.
    If labels are flattened, reshape them to (n_samples, n_labels).
    """
    if hasattr(y, "toarray"):
        y = y.toarray()

    y = np.asarray(y, dtype=int)

    if y.ndim == 1 and n_labels is not None and y.size % n_labels == 0:
        y = y.reshape(-1, n_labels)

    return y


def _idx_to_array(idx):
    """
    Convert index output from iterative_train_test_split to a flat integer array.
    """
    if hasattr(idx, "toarray"):
        idx = idx.toarray()

    idx = np.asarray(idx)

    # In case a boolean mask is returned.
    if idx.dtype == bool:
        return np.where(idx.ravel())[0]

    return idx.ravel().astype(int)


def _to_numpy(obj):
    if hasattr(obj, "toarray"):
        return obj.toarray()
    return np.asarray(obj)


def _looks_like_labels(obj, n_labels, n_total):
    """
    Heuristic to decide whether an object returned by
    iterative_train_test_split is a label matrix or an index matrix.
    """
    arr = _to_numpy(obj)

    # Normal case: labels have shape (n_samples, n_labels).
    if arr.ndim == 2:
        return arr.shape[1] == n_labels

    # Some versions may return flattened labels.
    if arr.ndim == 1:
        if arr.size == 0:
            return False

        # Labels are binary 0/1.
        # Index arrays contain values from 0 to n_total - 1.
        unique = np.unique(arr)
        values_are_binary = (
            len(unique) <= 2 and set(unique.tolist()).issubset({0, 1})
        )

        if values_are_binary and arr.size % n_labels == 0:
            return True

    return False


def split_train_dev(df: pd.DataFrame, dev_size: float = DEV_SIZE, seed: int = SEED):
    """
    Split training data into train/dev.

    Uses iterative multi-label stratification if scikit-multilearn is installed.
    This version is robust to different return orders from
    iterative_train_test_split.
    """
    texts = df[TEXT_COL].astype(str).to_numpy()
    y = (df[LABEL_COLS].to_numpy() > 0).astype(int)

    n_labels = y.shape[1]
    n_total = len(texts)
    indices = np.arange(n_total).reshape(-1, 1)

    if iterative_train_test_split is not None:
        try:
            raw_outputs = iterative_train_test_split(
                indices,
                y,
                test_size=dev_size,
            )

            index_objs = []
            label_objs = []

            for obj in raw_outputs:
                if _looks_like_labels(obj, n_labels, n_total):
                    label_objs.append(obj)
                else:
                    index_objs.append(obj)

            if len(index_objs) == 2 and len(label_objs) == 2:
                idx_train = _idx_to_array(index_objs[0])
                idx_dev = _idx_to_array(index_objs[1])

                y_train = _to_dense_labels(label_objs[0], n_labels)
                y_dev = _to_dense_labels(label_objs[1], n_labels)

                # If label order is accidentally reversed, try to align by row count.
                if len(idx_train) != y_train.shape[0] and len(idx_train) == y_dev.shape[0]:
                    y_train, y_dev = y_dev, y_train

                if len(idx_dev) != y_dev.shape[0] and len(idx_dev) == y_train.shape[0]:
                    y_train, y_dev = y_dev, y_train

                if len(idx_train) == y_train.shape[0] and len(idx_dev) == y_dev.shape[0]:
                    return (
                        texts[idx_train].tolist(),
                        texts[idx_dev].tolist(),
                        y_train,
                        y_dev,
                    )
                else:
                    print("Iterative stratification returned inconsistent shapes. Falling back.")

        except Exception as e:
            print("Iterative stratification failed. Falling back to standard splitting.")
            print(e)

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
# Text formatting
# ============================================================

def build_text(question: str) -> str:
    """
    Qwen2.5-14B-Instruct is an instruction model, so we give it a short
    classification instruction. The classification head then predicts
    the multi-label targets.
    """
    return (
        "Classify the required task labels for the following tax/legal question. "
        f"Available labels: {LABEL_TEXT}.\n"
        f"Question: {question}"
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
    labels = (np.asarray(labels) > 0).astype(int)

    probs = sigmoid(logits)
    thresholds = np.zeros(labels.shape[1], dtype=float)

    for label_idx in range(labels.shape[1]):
        col = labels[:, label_idx]
        s = probs[:, label_idx]
        s = np.nan_to_num(s, nan=-1e9)

        if len(s) == 0:
            thresholds[label_idx] = 0.5
            continue

        if col.sum() == 0:
            thresholds[label_idx] = 1.01
            continue

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
        )
    )

    print(f"Hamming loss: {hamming_loss(labels, preds):.4f}")
    print(f"Exact match ratio: {accuracy_score(labels, preds):.4f}")

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
    y = (np.asarray(y) > 0).astype(int)

    pos_counts = y.sum(axis=0).astype(np.float32)
    neg_counts = len(y) - pos_counts

    weights = np.ones_like(pos_counts)

    positive_mask = pos_counts > 0
    weights[positive_mask] = neg_counts[positive_mask] / pos_counts[positive_mask]

    weights = np.clip(weights, 1.0, clip)

    return torch.tensor(weights, dtype=torch.float)


# ============================================================
# Data collator and trainer
# ============================================================

class MultiLabelDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        labels = [feature.pop("labels", None) for feature in features]
        batch = super().__call__(features)

        if labels and labels[0] is not None:
            labels_tensor = torch.tensor(labels, dtype=torch.float)
            labels_tensor = (labels_tensor > 0).float()
            batch["labels"] = labels_tensor

        return batch


class WeightedBCETrainer(Trainer):
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

    if include_true_labels:
        keep_cols.extend([col for col in LABEL_COLS if col in out.columns])

    keep_cols = list(dict.fromkeys(keep_cols))

    out[keep_cols].to_csv(path, index=False)
    print(f"Saved predictions: {path}")


# ============================================================
# PEFT helper
# ============================================================

def get_head_module_names(model):
    """
    Detect the classification head module name.
    Qwen sequence classification usually uses `score`, but some models use `classifier`.
    """
    candidates = set()

    for name, module in model.named_modules():
        if name.endswith("score") or name.endswith("classifier"):
            candidates.add(name.split(".")[-1])

    if not candidates:
        candidates = {"score"}

    return list(candidates)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    setup_output_logging()

    train_path = find_train_path()

    print("=" * 90)
    print("Qwen2.5-14B-Instruct multi-label LoRA classifier")
    print("=" * 90)

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"BF16 supported: {cuda_bf16_supported()}")

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

    # ========================================================
    # Tokenizer
    # ========================================================
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Decoder-only models should use left padding for classification/generation.
    tokenizer.padding_side = "left"

    def tokenize_function(batch):
        texts = [build_text(question) for question in batch["text"]]
        return tokenizer(
            texts,
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
    # Precision and quantization setup
    # ========================================================
    use_bf16 = cuda_bf16_supported()

    if use_bf16:
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    bnb_config = None

    if USE_QLORA:
        print("Using QLoRA 4-bit quantization.")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        print("Using BF16 LoRA without 4-bit quantization.")

    # ========================================================
    # Model
    # ========================================================
    model_from_pretrained_kwargs = dict(
        num_labels=len(LABEL_COLS),
        problem_type="multi_label_classification",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="auto",
    )

    if bnb_config is not None:
        model_from_pretrained_kwargs["quantization_config"] = bnb_config

    # Newer Transformers versions prefer `dtype`, older ones use `torch_dtype`.
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            dtype=torch_dtype,
            **model_from_pretrained_kwargs,
        )
    except TypeError:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch_dtype,
            **model_from_pretrained_kwargs,
        )

    model.config.id2label = {i: label for i, label in enumerate(LABEL_COLS)}
    model.config.label2id = {label: i for i, label in enumerate(LABEL_COLS)}

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model.config.use_cache = False

    # ========================================================
    # LoRA
    # ========================================================
    head_modules = get_head_module_names(model)
    print(f"Classification head modules to save/train: {head_modules}")

    if USE_QLORA:
        model = prepare_model_for_kbit_training(model)

    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        target_modules=target_modules,
        modules_to_save=head_modules,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Needed for gradient checkpointing with PEFT.
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        try:
            model.get_input_embeddings().weight.requires_grad = True
        except Exception:
            pass

    # ========================================================
    # Training arguments
    # ========================================================
    training_arg_names = get_training_argument_names()
    eval_strategy_key = get_eval_strategy_argument_name(training_arg_names)

    training_kwargs = dict(
        output_dir=str(OUTPUT_DIR),
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        save_strategy="epoch",
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="ap_macro",
        greater_is_better=True,
        gradient_checkpointing=True,
        bf16=use_bf16,
        fp16=False,
        report_to="none",
        seed=SEED,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        optim="adamw_torch",
    )

    training_kwargs[eval_strategy_key] = "epoch"

    # Filter arguments for compatibility with different Transformers versions.
    if training_arg_names:
        training_kwargs = {
            key: value
            for key, value in training_kwargs.items()
            if key in training_arg_names
        }

        training_kwargs[eval_strategy_key] = "epoch"

        if "bf16" in training_arg_names:
            training_kwargs["bf16"] = use_bf16

        if "fp16" in training_arg_names:
            training_kwargs["fp16"] = False

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

    print("\nStarting Qwen2.5-14B-Instruct LoRA training...")
    trainer.train()

    # ========================================================
    # Threshold tuning on dev
    # ========================================================
    print("\nTuning thresholds on dev set...")
    dev_output = trainer.predict(dev_ds)
    dev_logits = np.asarray(dev_output.predictions)
    dev_labels = np.asarray(dev_output.label_ids)

    thresholds = optimize_thresholds(dev_logits, dev_labels)

    print("\nDev-tuned thresholds:")
    for label_name, threshold in zip(LABEL_COLS, thresholds):
        print(f"{label_name:25s} {threshold:.4f}")

    report_with_threshold(
        dev_labels,
        dev_logits,
        thresholds,
        title="Qwen2.5-14B dev report using thresholds tuned on dev",
    )

    # ========================================================
    # External test evaluation
    # ========================================================
    print("\nEvaluating external test set using dev-tuned thresholds...")
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
            title="Qwen2.5-14B test report using dev-tuned thresholds",
        )
    else:
        print("Test labels not found. Saving predictions without test metrics.")

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
        PREDICTION_DIR / "test_qwen2_5_14b_predictions.csv",
        include_true_labels=test_has_labels,
    )
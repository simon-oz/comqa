# src/ensemble_scores.py

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import load
from sentence_transformers import SentenceTransformer
from sklearn.base import clone
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
)
from peft import PeftModel


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

TRAIN_CANDIDATES = [
    "data/train_dataset.txt",
    "train_dataset.txt",
    "data/train_dataset.jsonl",
    "train_dataset.jsonl",
]

TEST_PATH = "data/test_dataset.jsonl"

DEV_SIZE = 0.20
SEED = 42

# ------------------------------------------------------------
# Model availability
# ------------------------------------------------------------

ENABLE_LOGREG = True
ENABLE_SVM = True
ENABLE_DEBERTA = True
ENABLE_QWEN = True

LOGREG_MODEL_PATH = Path("artifacts/logreg_ovr.joblib")
LOGREG_SCALER_PATH = Path("artifacts/logreg_scaler.joblib")

SVM_MODEL_PATH = Path("artifacts/svm_ovr.joblib")
SVM_SCALER_PATH = Path("artifacts/svm_scaler.joblib")

DEBERTA_MODEL_DIR = Path("deberta_large_multilabel/final")

QWEN_BASE_MODEL = "Qwen/Qwen2.5-14B-Instruct"
QWEN_ADAPTER_DIR = Path("qwen2_5_14b_multilabel/final")

# Set this to True only if the Qwen model was trained with QLoRA 4-bit.
QWEN_USE_QLORA = False

# ------------------------------------------------------------
# IMPORTANT:
# Set these to the exact embedding models used when training
# Logistic Regression and SVM.
#
# For example, if you trained Logistic Regression using
# BAAI/bge-base-en-v1.5, set it here.
# ------------------------------------------------------------

CLASSICAL_MODELS = {
    "logreg": {
        "model_path": LOGREG_MODEL_PATH,
        "scaler_path": LOGREG_SCALER_PATH,
        "embedding_model": "intfloat/e5-large-v2",
        "prefer_proba": True,
    },
    "svm": {
        "model_path": SVM_MODEL_PATH,
        "scaler_path": SVM_SCALER_PATH,
        "embedding_model": "intfloat/e5-large-v2",
        "prefer_proba": False,
    },
}

# ------------------------------------------------------------
# Ensemble settings
# ------------------------------------------------------------

# Starting weights. These are only used as candidates.
# The script can search for better weights on dev.
MODEL_WEIGHTS = {
    "logreg": 1.0,
    "svm": 1.0,
    "deberta": 1.0,
    "qwen": 1.0,
}

# Weight search settings.
SEARCH_WEIGHTS = True
WEIGHT_SEARCH_METRIC = "ap_macro"  # "ap_macro" or "f1_macro"
N_WEIGHT_TRIALS = 300

# Threshold settings.
BOOTSTRAP_THRESHOLDS = True
N_THRESHOLD_BOOTSTRAP = 200

# Neural inference settings.
DEBERTA_BATCH_SIZE = 8
QWEN_BATCH_SIZE = 2
MAX_LENGTH = 512

OUTPUT_DIR = Path("artifacts")
PREDICTION_DIR = Path("predictions")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

ENSEMBLE_WEIGHTS_PATH = OUTPUT_DIR / "ensemble_score_weights.json"
ENSEMBLE_THRESHOLDS_PATH = OUTPUT_DIR / "ensemble_score_thresholds.npy"
ENSEMBLE_PREDICTIONS_PATH = PREDICTION_DIR / "test_ensemble_score_predictions.csv"


# ============================================================
# Safe one-vs-rest classifier
#
# This class is needed so joblib can load the saved classical
# models if they were saved using the same class.
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

    def scores(self, X, prefer_proba: bool = False):
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

    for col in LABEL_COLS:
        if col not in df.columns:
            df[col] = 0

        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df[col] = (df[col] > 0).astype(int)

    df = df.dropna(subset=[TEXT_COL]).reset_index(drop=True)

    return df, has_labels


def split_train_dev(df: pd.DataFrame, dev_size: float = DEV_SIZE, seed: int = SEED):
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
# Metrics and thresholds
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


def bootstrap_optimize_thresholds(
    y_true: np.ndarray,
    scores: np.ndarray,
    n_bootstrap: int = N_THRESHOLD_BOOTSTRAP,
    seed: int = SEED,
) -> np.ndarray:
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    rng = np.random.default_rng(seed)
    n = len(y_true)

    all_thresholds = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_y = y_true[idx]
        boot_scores = scores[idx]

        thresholds = optimize_thresholds(boot_y, boot_scores)
        all_thresholds.append(thresholds)

    all_thresholds = np.array(all_thresholds)

    return np.median(all_thresholds, axis=0)


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
            digits=4
        )
    )

    print(f"Hamming loss: {hamming_loss(y_true, y_pred):.4f}")
    print(f"Exact match ratio: {accuracy_score(y_true, y_pred):.4f}")

    print(f"F1 micro: {f1_score(y_true, y_pred, average='micro', zero_division=0):.4f}")
    print(f"F1 macro: {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"F1 weighted: {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"F1 samples: {f1_score(y_true, y_pred, average='samples', zero_division=0):.4f}")


# ============================================================
# Rank normalization
# ============================================================

def fit_rank_transformer(dev_scores: np.ndarray):
    """
    Fit per-label rank transformers using dev scores.
    """
    dev_scores = np.nan_to_num(np.asarray(dev_scores, dtype=float), nan=-1e9)

    transformers = []

    for label_idx in range(dev_scores.shape[1]):
        col = dev_scores[:, label_idx]
        sorted_col = np.sort(col)

        transformers.append(sorted_col)

    return transformers


def apply_rank_transform(scores: np.ndarray, transformers) -> np.ndarray:
    """
    Transform scores to dev-based percentile ranks in [0, 1].
    """
    scores = np.nan_to_num(np.asarray(scores, dtype=float), nan=-1e9)

    out = np.zeros_like(scores, dtype=float)

    for label_idx, sorted_col in enumerate(transformers):
        s = scores[:, label_idx]

        if len(sorted_col) == 0:
            out[:, label_idx] = 0.5
            continue

        min_val = sorted_col[0]
        max_val = sorted_col[-1]

        if np.isclose(min_val, max_val):
            out[:, label_idx] = 0.5
            continue

        out[:, label_idx] = np.searchsorted(sorted_col, s, side="right") / float(len(sorted_col))

    return np.clip(out, 0.0, 1.0)


# ============================================================
# Classical model scoring
# ============================================================

_EMBEDDER_CACHE = {}
_EMBEDDING_CACHE = {}


def get_embedder(model_name: str):
    if model_name not in _EMBEDDER_CACHE:
        _EMBEDDER_CACHE[model_name] = SentenceTransformer(model_name)
    return _EMBEDDER_CACHE[model_name]


def get_embeddings(model_name: str, texts, cache_key: str):
    key = (model_name, cache_key)

    if key not in _EMBEDDING_CACHE:
        embedder = get_embedder(model_name)

        _EMBEDDING_CACHE[key] = embedder.encode(
            texts,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    return _EMBEDDING_CACHE[key]


def get_model_scores(model, X, prefer_proba: bool = False):
    if hasattr(model, "scores"):
        return model.scores(X, prefer_proba=prefer_proba)

    # Fallback, in case the saved object has estimators but no scores method.
    n_samples = X.shape[0]
    score_columns = []

    estimators = getattr(model, "estimators_", [])
    constant_labels = getattr(model, "constant_labels_", [None] * len(estimators))

    for est, constant_label in zip(estimators, constant_labels):
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
            s = raw[:, -1]

        elif hasattr(est, "decision_function"):
            raw = np.asarray(est.decision_function(X))

            if raw.ndim == 1:
                s = raw
            else:
                s = raw[:, -1]

        elif hasattr(est, "predict_proba"):
            raw = np.asarray(est.predict_proba(X))
            s = raw[:, -1]

        else:
            s = est.predict(X).astype(float)

        score_columns.append(np.asarray(s, dtype=float).ravel())

    return np.column_stack(score_columns)


def load_classical_scores(
    model_name: str,
    config: dict,
    dev_texts,
    test_texts,
):
    model_path = Path(config["model_path"])
    scaler_path = Path(config["scaler_path"])
    embedding_model = config["embedding_model"]
    prefer_proba = bool(config.get("prefer_proba", False))

    if not model_path.exists():
        print(f"Skipping {model_name}: model file not found at {model_path}")
        return None

    print(f"Loading {model_name} from {model_path}")
    model = load(model_path)

    scaler = None
    if scaler_path.exists():
        scaler = load(scaler_path)
    else:
        print(f"Warning: scaler not found for {model_name} at {scaler_path}")

    print(f"Embedding texts for {model_name} using {embedding_model}")

    X_dev = get_embeddings(embedding_model, dev_texts, cache_key=f"{model_name}_dev")
    X_test = get_embeddings(embedding_model, test_texts, cache_key=f"{model_name}_test")

    if scaler is not None:
        X_dev = scaler.transform(X_dev)
        X_test = scaler.transform(X_test)

    dev_scores = get_model_scores(model, X_dev, prefer_proba=prefer_proba)
    test_scores = get_model_scores(model, X_test, prefer_proba=prefer_proba)

    return dev_scores, test_scores


# ============================================================
# Hugging Face model scoring
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


def get_model_device(model):
    try:
        return next(model.parameters()).device
    except Exception:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_logits(
    model,
    tokenizer,
    texts,
    batch_size: int,
    max_length: int = MAX_LENGTH,
    prompt_fn=None,
):
    device = get_model_device(model)

    all_logits = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]

        if prompt_fn is not None:
            batch_texts = [prompt_fn(text) for text in batch_texts]

        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits.float().cpu().numpy()
        all_logits.append(logits)

    return np.vstack(all_logits)


def load_deberta_scores(dev_texts, test_texts):
    if not DEBERTA_MODEL_DIR.exists():
        print(f"Skipping DeBERTa: model directory not found at {DEBERTA_MODEL_DIR}")
        return None

    print(f"Loading DeBERTa from {DEBERTA_MODEL_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(DEBERTA_MODEL_DIR)

    torch_dtype = torch.bfloat16 if cuda_bf16_supported() else torch.float32

    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            DEBERTA_MODEL_DIR,
            dtype=torch_dtype,
            device_map="auto",
        )
    except TypeError:
        model = AutoModelForSequenceClassification.from_pretrained(
            DEBERTA_MODEL_DIR,
            torch_dtype=torch_dtype,
            device_map="auto",
        )

    model.eval()

    print("Predicting DeBERTa dev logits...")
    dev_logits = predict_logits(
        model=model,
        tokenizer=tokenizer,
        texts=dev_texts,
        batch_size=DEBERTA_BATCH_SIZE,
        max_length=MAX_LENGTH,
        prompt_fn=None,
    )

    print("Predicting DeBERTa test logits...")
    test_logits = predict_logits(
        model=model,
        tokenizer=tokenizer,
        texts=test_texts,
        batch_size=DEBERTA_BATCH_SIZE,
        max_length=MAX_LENGTH,
        prompt_fn=None,
    )

    del model
    del tokenizer
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return dev_logits, test_logits


def build_qwen_text(question: str) -> str:
    return (
        "Classify the required task labels for the following tax/legal question. "
        f"Available labels: {LABEL_TEXT}.\n"
        f"Question: {question}"
    )


def load_qwen_scores(dev_texts, test_texts):
    if not QWEN_ADAPTER_DIR.exists():
        print(f"Skipping Qwen: adapter directory not found at {QWEN_ADAPTER_DIR}")
        return None

    print(f"Loading Qwen tokenizer from {QWEN_ADAPTER_DIR}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            QWEN_ADAPTER_DIR,
            trust_remote_code=True,
        )
    except Exception:
        print("Falling back to base Qwen tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(
            QWEN_BASE_MODEL,
            trust_remote_code=True,
        )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    bnb_config = None

    if QWEN_USE_QLORA:
        print("Loading Qwen base model with 4-bit quantization.")
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

    torch_dtype = torch.bfloat16 if cuda_bf16_supported() else torch.float32

    print("Loading Qwen base model...")

    try:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            QWEN_BASE_MODEL,
            dtype=torch_dtype,
            **model_kwargs,
        )
    except TypeError:
        base_model = AutoModelForSequenceClassification.from_pretrained(
            QWEN_BASE_MODEL,
            torch_dtype=torch_dtype,
            **model_kwargs,
        )

    if base_model.config.pad_token_id is None:
        base_model.config.pad_token_id = tokenizer.pad_token_id

    print(f"Loading Qwen LoRA adapter from {QWEN_ADAPTER_DIR}")
    model = PeftModel.from_pretrained(
        base_model,
        QWEN_ADAPTER_DIR,
    )

    model.eval()

    print("Predicting Qwen dev logits...")
    dev_logits = predict_logits(
        model=model,
        tokenizer=tokenizer,
        texts=dev_texts,
        batch_size=QWEN_BATCH_SIZE,
        max_length=MAX_LENGTH,
        prompt_fn=build_qwen_text,
    )

    print("Predicting Qwen test logits...")
    test_logits = predict_logits(
        model=model,
        tokenizer=tokenizer,
        texts=test_texts,
        batch_size=QWEN_BATCH_SIZE,
        max_length=MAX_LENGTH,
        prompt_fn=build_qwen_text,
    )

    del model
    del base_model
    del tokenizer
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return dev_logits, test_logits


# ============================================================
# Ensemble combination
# ============================================================

def combine_score_dict(score_dict, weights):
    model_names = list(score_dict.keys())

    weight_values = np.array([float(weights[name]) for name in model_names], dtype=float)

    if weight_values.sum() <= 0:
        weight_values = np.ones_like(weight_values)

    weight_values = weight_values / weight_values.sum()

    stacked = np.stack([score_dict[name] for name in model_names], axis=0)
    combined = np.tensordot(weight_values, stacked, axes=(0, 0))

    return combined, weight_values


def search_ensemble_weights(
    dev_score_dict,
    y_dev,
    metric: str = WEIGHT_SEARCH_METRIC,
    n_trials: int = N_WEIGHT_TRIALS,
    seed: int = SEED,
):
    model_names = list(dev_score_dict.keys())
    n_models = len(model_names)

    if n_models == 1:
        return {model_names[0]: 1.0}, 1.0

    rng = np.random.default_rng(seed)

    stacked_dev = np.stack([dev_score_dict[name] for name in model_names], axis=0)

    candidates = []

    # Equal weights.
    candidates.append(np.ones(n_models) / n_models)

    # User-provided weights.
    user_weights = np.array([float(MODEL_WEIGHTS.get(name, 1.0)) for name in model_names], dtype=float)
    if user_weights.sum() > 0:
        candidates.append(user_weights / user_weights.sum())

    # Random Dirichlet weights.
    if SEARCH_WEIGHTS:
        for _ in range(n_trials):
            # Concentration 2.0 gives moderate weights.
            # Use 1.0 for more aggressive exploration.
            w = rng.dirichlet(np.ones(n_models) * 2.0)
            candidates.append(w)

    best_score = -np.inf
    best_weights = candidates[0]

    for weights in candidates:
        combined = np.tensordot(weights, stacked_dev, axes=(0, 0))

        if metric == "f1_macro":
            thresholds = optimize_thresholds(y_dev, combined)
            preds = (combined >= thresholds).astype(int)
            score = f1_score(y_dev, preds, average="macro", zero_division=0)
        else:
            score = safe_average_precision(y_dev, combined, average="macro")

        if np.isfinite(score) and score > best_score:
            best_score = score
            best_weights = weights

    best_weight_dict = {
        name: float(weight)
        for name, weight in zip(model_names, best_weights)
    }

    return best_weight_dict, float(best_score)


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 90)
    print("Score-level multi-model ensemble")
    print("=" * 90)

    train_path = find_train_path()

    print(f"Loading training data from: {train_path}")
    train_df, train_has_labels = load_jsonl(train_path)

    if not train_has_labels:
        print("Warning: some label columns are missing in the training dataset.")

    _, dev_texts, _, y_dev = split_train_dev(train_df)

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
        print("Test metrics will be skipped.")

    # --------------------------------------------------------
    # Collect raw scores from each model.
    # --------------------------------------------------------
    raw_dev_scores = {}
    raw_test_scores = {}

    # Classical models.
    for model_name, config in CLASSICAL_MODELS.items():
        if model_name == "logreg" and not ENABLE_LOGREG:
            continue

        if model_name == "svm" and not ENABLE_SVM:
            continue

        scores = load_classical_scores(
            model_name=model_name,
            config=config,
            dev_texts=dev_texts,
            test_texts=test_texts,
        )

        if scores is None:
            continue

        dev_scores, test_scores = scores
        raw_dev_scores[model_name] = dev_scores
        raw_test_scores[model_name] = test_scores

    # DeBERTa.
    if ENABLE_DEBERTA:
        scores = load_deberta_scores(dev_texts, test_texts)

        if scores is not None:
            dev_scores, test_scores = scores
            raw_dev_scores["deberta"] = dev_scores
            raw_test_scores["deberta"] = test_scores

    # Qwen.
    if ENABLE_QWEN:
        scores = load_qwen_scores(dev_texts, test_texts)

        if scores is not None:
            dev_scores, test_scores = scores
            raw_dev_scores["qwen"] = dev_scores
            raw_test_scores["qwen"] = test_scores

    if not raw_dev_scores:
        raise RuntimeError("No models were available for ensembling.")

    print("\nModels included in score-level ensemble:")
    for name in raw_dev_scores.keys():
        print(f"  - {name}")

    # --------------------------------------------------------
    # Rank-normalize scores.
    # --------------------------------------------------------
    print("\nRank-normalizing scores using dev set...")

    rank_dev_scores = {}
    rank_test_scores = {}

    for model_name in raw_dev_scores.keys():
        transformer = fit_rank_transformer(raw_dev_scores[model_name])

        rank_dev_scores[model_name] = apply_rank_transform(
            raw_dev_scores[model_name],
            transformer,
        )

        rank_test_scores[model_name] = apply_rank_transform(
            raw_test_scores[model_name],
            transformer,
        )

    # --------------------------------------------------------
    # Optional: print individual AP macro.
    # --------------------------------------------------------
    print("\nIndividual model AP macro after rank normalization:")
    for model_name in rank_dev_scores.keys():
        dev_ap = safe_average_precision(y_dev, rank_dev_scores[model_name], average="macro")

        if test_has_labels:
            test_ap = safe_average_precision(y_test, rank_test_scores[model_name], average="macro")
            print(f"  {model_name:10s} dev AP macro: {dev_ap:.4f} | test AP macro: {test_ap:.4f}")
        else:
            print(f"  {model_name:10s} dev AP macro: {dev_ap:.4f}")

    # --------------------------------------------------------
    # Search ensemble weights.
    # --------------------------------------------------------
    if SEARCH_WEIGHTS:
        print("\nSearching ensemble weights on dev set...")
        weights, search_score = search_ensemble_weights(
            dev_score_dict=rank_dev_scores,
            y_dev=y_dev,
            metric=WEIGHT_SEARCH_METRIC,
            n_trials=N_WEIGHT_TRIALS,
            seed=SEED,
        )

        print(f"Weight search metric: {WEIGHT_SEARCH_METRIC}")
        print(f"Best dev search score: {search_score:.4f}")
    else:
        model_names = list(rank_dev_scores.keys())
        total = sum(float(MODEL_WEIGHTS.get(name, 1.0)) for name in model_names)

        if total <= 0:
            weights = {name: 1.0 / len(model_names) for name in model_names}
        else:
            weights = {
                name: float(MODEL_WEIGHTS.get(name, 1.0)) / total
                for name in model_names
            }

    print("\nFinal ensemble weights:")
    for name, weight in weights.items():
        print(f"  {name:10s} {weight:.4f}")

    # --------------------------------------------------------
    # Combine scores.
    # --------------------------------------------------------
    combined_dev_scores, normalized_weights = combine_score_dict(rank_dev_scores, weights)
    combined_test_scores, _ = combine_score_dict(rank_test_scores, weights)

    # Update weights dict with normalized values.
    weights = {
        name: float(weight)
        for name, weight in zip(rank_dev_scores.keys(), normalized_weights)
    }

    # --------------------------------------------------------
    # Tune thresholds on dev.
    # --------------------------------------------------------
    print("\nTuning ensemble thresholds on dev set...")

    if BOOTSTRAP_THRESHOLDS:
        thresholds = bootstrap_optimize_thresholds(
            y_dev,
            combined_dev_scores,
            n_bootstrap=N_THRESHOLD_BOOTSTRAP,
            seed=SEED,
        )
    else:
        thresholds = optimize_thresholds(y_dev, combined_dev_scores)

    print("\nEnsemble thresholds:")
    for label_name, threshold in zip(LABEL_COLS, thresholds):
        print(f"  {label_name:25s} {threshold:.4f}")

    # --------------------------------------------------------
    # Evaluate.
    # --------------------------------------------------------
    dev_preds = (combined_dev_scores >= thresholds).astype(int)

    evaluate_predictions(
        y_dev,
        dev_preds,
        title="Score-level ensemble: dev report",
    )

    test_preds = (combined_test_scores >= thresholds).astype(int)

    if test_has_labels:
        evaluate_predictions(
            y_test,
            test_preds,
            title="Score-level ensemble: test report",
        )

    # --------------------------------------------------------
    # Save outputs.
    # --------------------------------------------------------
    with open(ENSEMBLE_WEIGHTS_PATH, "w", encoding="utf-8") as f:
        json.dump(weights, f, indent=2)

    np.save(ENSEMBLE_THRESHOLDS_PATH, thresholds)

    output_data = {}

    if "id" in test_df.columns:
        output_data["id"] = test_df["id"].tolist()

    if TEXT_COL in test_df.columns:
        output_data[TEXT_COL] = test_df[TEXT_COL].tolist()

    for i, col in enumerate(LABEL_COLS):
        output_data[f"pred_{col}"] = test_preds[:, i]
        output_data[f"ensemble_score_{col}"] = combined_test_scores[:, i]

    if test_has_labels:
        for i, col in enumerate(LABEL_COLS):
            output_data[col] = y_test[:, i]

    output_df = pd.DataFrame(output_data)
    output_df.to_csv(ENSEMBLE_PREDICTIONS_PATH, index=False)

    print("\nSaved ensemble weights:")
    print(ENSEMBLE_WEIGHTS_PATH)

    print("Saved ensemble thresholds:")
    print(ENSEMBLE_THRESHOLDS_PATH)

    print("Saved ensemble predictions:")
    print(ENSEMBLE_PREDICTIONS_PATH)


if __name__ == "__main__":
    main()
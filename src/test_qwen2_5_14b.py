# src/test_qwen2_5_14b.py
#
# Zero-shot test script for Qwen2.5-14B-Instruct.
# Uses ONLY:
#   - data/test_dataset.jsonl
#   - pretrained Qwen2.5-14B-Instruct
#
# Does NOT use:
#   - training data
#   - dev data
#   - fine-tuned adapter
#   - saved thresholds

import atexit
import json
import re
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
    classification_report,
    f1_score,
    hamming_loss,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
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

LABEL_TO_IDX = {label: i for i, label in enumerate(LABEL_COLS)}

TEST_PATH = Path("data/test_dataset.jsonl")

MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"

# Set True if you run out of GPU memory.
USE_4BIT = False

MAX_PROMPT_LENGTH = 1024
MAX_NEW_TOKENS = 192

# Optional: set to an integer for quick debugging, e.g. 5.
# Use None to evaluate the full test set.
MAX_TEST_SAMPLES = None

OUTPUT_DIR = Path("qwen2_5_14b_zero_shot")
LOG_DIR = Path("logs")
PREDICTION_DIR = Path("predictions")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PREDICTIONS_PATH = PREDICTION_DIR / "test_qwen2_5_14b_zero_shot_predictions.csv"
OUTPUT_RAW_PATH = OUTPUT_DIR / "test_qwen2_5_14b_zero_shot_raw_outputs.json"
OUTPUT_SUMMARY_PATH = OUTPUT_DIR / "test_qwen2_5_14b_zero_shot_summary.json"
OUTPUT_TIMING_PATH = OUTPUT_DIR / "test_qwen2_5_14b_zero_shot_timing.json"


# ============================================================
# Label definitions for zero-shot prompting
# ============================================================

LABEL_DEFINITIONS = {
    "retrieval": "retrieval of legal or tax rules, provisions, rulings, rates, or external factual information",
    "comparison": "comparison of two or more items, regimes, treatments, structures, periods, or options",
    "aggregation": "collection, listing, combination, or organisation of multiple rules, obligations, items, or categories",
    "context_resolution": "resolution of references to supplied context, earlier information, documents, entities, or prior assumptions",
    "temporal": "dependence on dates, periods, deadlines, historical law, current law, or law in force at a particular time",
    "legal_citation": "citation of legislation, cases, rulings, interpretations, or authoritative legal or tax sources",
    "evidence_extraction": "extraction of facts, figures, clauses, amounts, dates, parties, or evidence from documents or records",
    "multi_document": "use or reconciliation of multiple documents, sources, reports, invoices, statements, or contracts",
    "summary": "summarisation, condensation, or overview of obligations, findings, rules, risks, or documents",
    "analyze": "reasoning, interpretation, classification, reconciliation, or analysis of facts or rules",
    "evaluate": "judgment of risk, reliability, compliance, strength of evidence, exposure, validity, or confidence",
    "recommend": "recommendation of actions, steps, strategies, timing, controls, or next steps",
}


def build_label_block():
    lines = []
    for label in LABEL_COLS:
        lines.append(f"- {label}: {LABEL_DEFINITIONS[label]}")
    return "\n".join(lines)


LABEL_BLOCK = build_label_block()

SYSTEM_PROMPT = (
    "You are a strict multi-label classification system for tax/legal questions. "
    "You must choose zero or more labels from the allowed label set. "
    "Do not invent labels. Do not include explanations. "
    "Return only valid JSON in this format: {\"labels\": [\"label_name\", ...]}. "
    "If no label applies, return {\"labels\": []}."
)


def build_user_prompt(question: str) -> str:
    return (
        "Allowed labels:\n"
        f"{LABEL_BLOCK}\n\n"
        f"Question:\n{question}\n\n"
        "Task: Select all allowed labels that are required by this question.\n"
        "Return only valid JSON:\n"
        "{\"labels\": [\"label_name\", ...]}"
    )


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
    log_path = LOG_DIR / f"test_qwen2_5_14b_zero_shot_{timestamp}.log"

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

    # True labels are only used for final evaluation, not for inference.
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
# Model loading
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


def load_model_and_tokenizer():
    print(f"Loading tokenizer: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Decoder-only models usually work better with left padding for generation.
    tokenizer.padding_side = "left"

    print(f"Loading model: {MODEL_NAME}")

    if USE_4BIT:
        print("Using 4-bit quantization.")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if cuda_bf16_supported() else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        torch_dtype = torch.bfloat16 if cuda_bf16_supported() else torch.float32
        print(f"Using dtype: {torch_dtype}")

        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
            )
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch_dtype,
                device_map="auto",
                trust_remote_code=True,
            )

    model.eval()

    return model, tokenizer


# ============================================================
# Zero-shot generation
# ============================================================

def generate_zero_shot_response(model, tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(question)},
    ]

    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = (
            f"System: {SYSTEM_PROMPT}\n"
            f"User: {build_user_prompt(question)}\n"
            "Assistant:"
        )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_PROMPT_LENGTH,
    )

    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[1]

    generated_text = tokenizer.decode(
        output_ids[0][input_len:],
        skip_special_tokens=True,
    )

    return generated_text.strip()


# ============================================================
# Output parsing
# ============================================================

LABEL_ALIASES = {}

for label in LABEL_COLS:
    LABEL_ALIASES[label.lower()] = label
    LABEL_ALIASES[label] = label

    spaced = label.replace("_", " ")
    LABEL_ALIASES[spaced.lower()] = label
    LABEL_ALIASES[spaced] = label
    LABEL_ALIASES[spaced.replace(" ", "")] = label
    LABEL_ALIASES[spaced.replace(" ", "_")] = label

# Additional common aliases.
LABEL_ALIASES.update(
    {
        "multi doc": "multi_document",
        "multi docs": "multi_document",
        "multiple document": "multi_document",
        "multiple documents": "multi_document",
        "multi-document": "multi_document",
        "cross document": "multi_document",
        "cross documents": "multi_document",

        "legal citations": "legal_citation",
        "citation": "legal_citation",
        "citations": "legal_citation",
        "cite law": "legal_citation",
        "cite legislation": "legal_citation",
        "legislation citation": "legal_citation",

        "evidence": "evidence_extraction",
        "extract evidence": "evidence_extraction",
        "extraction": "evidence_extraction",
        "fact extraction": "evidence_extraction",
        "figure extraction": "evidence_extraction",

        "summarise": "summary",
        "summarize": "summary",
        "summarization": "summary",

        "analysis": "analyze",
        "analytical reasoning": "analyze",
        "reasoning": "analyze",
        "interpretation": "analyze",

        "evaluation": "evaluate",
        "risk assessment": "evaluate",
        "risk evaluation": "evaluate",
        "assessment": "evaluate",

        "recommendation": "recommend",
        "recommendations": "recommend",
        "advice": "recommend",
        "next steps": "recommend",

        "context": "context_resolution",
        "context resolution": "context_resolution",
        "coreference": "context_resolution",
        "reference resolution": "context_resolution",

        "time": "temporal",
        "timing": "temporal",
        "date": "temporal",
        "dates": "temporal",
        "historical": "temporal",
        "law in force": "temporal",
        "time dependent": "temporal",
    }
)


def normalize_label(value):
    if value is None:
        return None

    s = str(value).strip().lower()

    # Remove markdown/code markers and punctuation.
    s = s.strip("[]{}\"'` ")
    s = re.sub(r"[^a-z0-9_ ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    if not s:
        return None

    if s in LABEL_ALIASES:
        return LABEL_ALIASES[s]

    s_underscore = s.replace(" ", "_")
    if s_underscore in LABEL_ALIASES:
        return LABEL_ALIASES[s_underscore]

    s_spaced = s.replace("_", " ")
    if s_spaced in LABEL_ALIASES:
        return LABEL_ALIASES[s_spaced]

    # Substring fallback.
    for alias, canonical in LABEL_ALIASES.items():
        if alias and alias.lower() in s:
            return canonical

    return None


def clean_model_output(text: str) -> str:
    text = text.strip()

    # Remove code fences if present.
    text = re.sub(r"```json", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)

    return text.strip()


def extract_json_object(text: str):
    text = clean_model_output(text)

    # Try direct JSON.
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try extracting the first JSON object.
    start_obj = text.find("{")
    end_obj = text.rfind("}")

    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
        candidate = text[start_obj:end_obj + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    # Try extracting a JSON list.
    start_list = text.find("[")
    end_list = text.rfind("]")

    if start_list != -1 and end_list != -1 and end_list > start_list:
        candidate = text[start_list:end_list + 1]
        try:
            return {"labels": json.loads(candidate)}
        except Exception:
            pass

    return None


def _is_truthy(value):
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return float(value) > 0.0

    if isinstance(value, str):
        return value.strip().lower() in {
            "true",
            "yes",
            "y",
            "1",
            "included",
            "include",
            "selected",
            "select",
        }

    return False


def parse_labels_from_output(raw_output: str):
    obj = extract_json_object(raw_output)

    candidate_labels = None

    if isinstance(obj, list):
        candidate_labels = obj

    elif isinstance(obj, dict):
        # Preferred case: {"labels": [...]}
        label_keys = [
            key for key in obj.keys()
            if str(key).strip().lower() in {"labels", "label", "selected_labels", "categories"}
        ]

        if label_keys:
            candidate_labels = obj[label_keys[0]]
        else:
            # Alternative case: {"retrieval": true, "comparison": false, ...}
            candidate_labels = [
                key for key, value in obj.items()
                if _is_truthy(value)
            ]

    if candidate_labels is None:
        # Fallback: scan raw text for known aliases.
        cleaned = clean_model_output(raw_output).lower()
        parsed = set()

        for alias, canonical in LABEL_ALIASES.items():
            if alias and alias.lower() in cleaned:
                parsed.add(canonical)

        return sorted(parsed)

    if isinstance(candidate_labels, str):
        if "," in candidate_labels:
            candidate_labels = candidate_labels.split(",")
        else:
            candidate_labels = [candidate_labels]

    if not isinstance(candidate_labels, list):
        candidate_labels = []

    parsed = set()

    for item in candidate_labels:
        canonical = normalize_label(item)
        if canonical is not None:
            parsed.add(canonical)

    return sorted(parsed)


# ============================================================
# Evaluation
# ============================================================

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray):
    print("=" * 90)
    print("Qwen2.5-14B-Instruct zero-shot test report")
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

    return metrics


# ============================================================
# Prediction saving
# ============================================================

def to_serializable_scalar(value):
    """
    Convert pandas / numpy scalar values into JSON-serializable Python values.
    This fixes errors such as:
        AttributeError: 'str' object has no attribute 'item'
    """
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


def save_predictions(
    test_df: pd.DataFrame,
    preds: np.ndarray,
    raw_outputs: list,
    include_true_labels: bool = True,
):
    out = test_df.copy()

    for i, label in enumerate(LABEL_COLS):
        out[f"pred_{label}"] = preds[:, i]

    out["raw_zero_shot_output"] = raw_outputs

    keep_cols = []

    if "id" in out.columns:
        keep_cols.append("id")

    keep_cols.append(TEXT_COL)

    keep_cols.extend([f"pred_{label}" for label in LABEL_COLS])

    if include_true_labels:
        keep_cols.extend([label for label in LABEL_COLS if label in out.columns])

    keep_cols.append("raw_zero_shot_output")

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
    print("Qwen2.5-14B-Instruct zero-shot evaluation")
    print("=" * 90)

    print("Mode: zero-shot generative classification")
    print("Train data used: no")
    print("Dev data used: no")
    print("Fine-tuned adapter used: no")
    print("Saved thresholds used: no")

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"BF16 supported: {cuda_bf16_supported()}")

    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Test dataset not found: {TEST_PATH}")

    print(f"Loading test data from: {TEST_PATH}")
    test_df, test_has_labels = load_test_jsonl(TEST_PATH)

    questions = test_df[TEXT_COL].astype(str).tolist()
    y_true = (test_df[LABEL_COLS].to_numpy() > 0).astype(int)

    print(f"Test size: {len(test_df)}")

    if not test_has_labels:
        print("Warning: test dataset does not contain all label columns.")
        print("Metrics will be skipped.")

    model, tokenizer = load_model_and_tokenizer()

    preds = np.zeros((len(questions), len(LABEL_COLS)), dtype=int)
    raw_outputs = []

    print("\nRunning zero-shot inference...")

    inference_start_time = time.perf_counter()

    for idx, question in enumerate(questions):
        try:
            raw_output = generate_zero_shot_response(model, tokenizer, question)
        except Exception as e:
            raw_output = f"ERROR: {e}"

        raw_outputs.append(raw_output)

        predicted_labels = parse_labels_from_output(raw_output)

        for label in predicted_labels:
            if label in LABEL_TO_IDX:
                preds[idx, LABEL_TO_IDX[label]] = 1

        if (idx + 1) % 10 == 0 or (idx + 1) == len(questions):
            print(f"Processed {idx + 1}/{len(questions)} test examples.")

    inference_elapsed_time = time.perf_counter() - inference_start_time

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n" + "=" * 90)
    print("Inference time summary")
    print("=" * 90)
    print(f"Inference time: {inference_elapsed_time:.2f} seconds ({inference_elapsed_time / 60:.2f} minutes)")

    if test_has_labels:
        metrics = evaluate_predictions(y_true, preds)
    else:
        print("Test labels not found. Skipping metrics.")
        metrics = {}

    save_predictions(
        test_df,
        preds,
        raw_outputs,
        include_true_labels=test_has_labels,
    )

    raw_payload = {
        "model_name": MODEL_NAME,
        "mode": "zero-shot",
        "test_path": str(TEST_PATH),
        "test_size": int(len(test_df)),
        "max_test_samples": MAX_TEST_SAMPLES,
        "outputs": [
            {
                "id": to_serializable_scalar(test_df.loc[i, "id"]) if "id" in test_df.columns else None,
                "question": str(test_df.loc[i, TEXT_COL]),
                "raw_output": raw_outputs[i],
                "parsed_labels": [
                    LABEL_COLS[j] for j in range(len(LABEL_COLS)) if preds[i, j] == 1
                ],
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
        "mode": "zero-shot",
        "use_train_data": False,
        "use_dev_data": False,
        "use_fine_tuned_adapter": False,
        "use_saved_thresholds": False,
        "test_path": str(TEST_PATH),
        "test_size": int(len(test_df)),
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
    print(f"Saved predictions: {OUTPUT_PREDICTIONS_PATH}")
    print(f"Saved raw outputs: {OUTPUT_RAW_PATH}")
    print(f"Saved summary: {OUTPUT_SUMMARY_PATH}")
    print(f"Saved timing: {OUTPUT_TIMING_PATH}")


if __name__ == "__main__":
    setup_output_logging()

    try:
        main()
    finally:
        shutdown_output_logging()
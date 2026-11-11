"""Text-only scorers for base and fine-tuned DeBERTa/ModernBERT/Qwen checkpoints.

External sentence embeddings are intentionally not passed into these models.
Embedding influence is measured separately via explicitly labeled score fusion.
"""

import gc
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np

from embedding_experiment_utils import LABEL_COLS

# CHANGE 3a: register ModernBERT (base + fine-tuned) and split out MLM classifiers
# so that zero-shot routing (predict_mlm, cloze metadata, fixed thresholds) applies
# to both DeBERTa and ModernBERT without duplicating `name == "deberta"` checks.
NEURAL_CLASSIFIERS = (
    "deberta", "modernbert", "qwen",
    "deberta_finetuned", "modernbert_finetuned", "qwen_finetuned",
)
# Bidirectional encoders routed through predict_mlm for zero-shot.
MLM_CLASSIFIERS = frozenset(("deberta", "modernbert"))

VERBALIZERS = [
    "retrieval", "comparison", "aggregation", "context", "time", "citation",
    "extraction", "documents", "summary", "analysis", "evaluation", "recommendation",
]
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
SYSTEM_PROMPT = (
    "You are a strict multi-label classification system for tax/legal questions. "
    "Choose zero or more labels from the allowed label set. Do not invent labels. "
    'Return only valid JSON: {"labels": ["label_name", ...]}. '
    'If no label applies, return {"labels": []}. Do not include explanations.'
)


def qwen_classification_text(question):
    # This format must match the existing sequence-classification adapter training.
    return (
        "Classify the required task labels for the following tax/legal question. "
        f"Available labels: {', '.join(LABEL_COLS)}.\nQuestion: {question}"
    )


def qwen_messages(question):
    block = "\n".join(f"- {label}: {LABEL_DEFINITIONS[label]}" for label in LABEL_COLS)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Allowed labels:\n{block}\n\nQuestion:\n{question}\n\n"
         'Select all required labels. Return only JSON: {"labels": ["label_name", ...]}.'},
    ]


# CHANGE 3e: Qwen zero-shot parsing now survives a single malformed row. The
# JSON path is unchanged (still strict, still rejects unknown labels). Only the
# json.JSONDecodeError case is new: it extracts label names that literally appear
# as whole words in the raw output. It never fabricates — an unmentioned label
# stays 0.
def parse_qwen_labels(raw):
    """Never turn malformed output or negative prose into fabricated predictions."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Fallback for a single malformed row: only accept labels that appear as
        # whole words. Anything not explicitly mentioned stays 0.
        lowered = text.lower()
        found = set()
        for label in LABEL_COLS:
            for variant in (label, label.replace("_", " ")):
                if re.search(rf"(?<![a-z0-9_]){re.escape(variant)}(?![a-z0-9_])", lowered):
                    found.add(label)
                    break
        return np.array([int(label in found) for label in LABEL_COLS], dtype=float)

    labels = obj.get("labels") if isinstance(obj, dict) else obj
    if not isinstance(labels, list) or any(not isinstance(label, str) for label in labels):
        raise ValueError("Qwen must return a JSON object with a labels list")
    labels = [label.strip().lower().replace(" ", "_") for label in labels]
    unknown = set(labels) - set(LABEL_COLS)
    if unknown:
        raise ValueError(f"Qwen returned unknown labels: {sorted(unknown)}")
    return np.array([int(label in labels) for label in LABEL_COLS], dtype=float)


def checkpoint_label_order(config, explicit_labels=None):
    """Align named checkpoint outputs; generic LABEL_0 mappings need an explicit order."""
    mapping = getattr(config, "id2label", {}) or {}
    labels = [mapping.get(i, mapping.get(str(i))) for i in range(len(LABEL_COLS))]
    if getattr(config, "num_labels", None) != len(LABEL_COLS):
        raise ValueError("Fine-tuned checkpoint must have exactly 12 output labels")
    if set(labels) == set(LABEL_COLS):
        return [labels.index(label) for label in LABEL_COLS]
    if explicit_labels is None:
        raise ValueError("Checkpoint label order is unknown. Supply --checkpoint-labels in training order")
    if len(explicit_labels) != len(LABEL_COLS) or set(explicit_labels) != set(LABEL_COLS):
        raise ValueError("--checkpoint-labels must contain each task label exactly once")
    return [explicit_labels.index(label) for label in LABEL_COLS]


def ensure_loaded_weights(info, *, allowed_missing=()):
    missing = [key for key in info.get("missing_keys", []) if key not in allowed_missing]
    if missing or info.get("mismatched_keys") or info.get("error_msgs"):
        raise ValueError(f"Checkpoint would use untrained/mismatched weights: {missing or info}")


def checkpoint_metadata(source):
    path = Path(source)
    metadata = {"checkpoint": str(path.resolve()) if path.is_dir() else str(source)}
    if path.is_dir():
        # Inventory large weight files without reading tens of GB just to log metadata.
        metadata["files"] = [
            {"name": str(file.relative_to(path)), "size": file.stat().st_size,
             "mtime_ns": file.stat().st_mtime_ns}
            for file in sorted(path.rglob("*")) if file.is_file()
        ]
        metadata["config_sha256"] = {
            file.name: hashlib.sha256(file.read_bytes()).hexdigest()
            for file in path.glob("*config.json")
        }
    return metadata


def predict_classification(model, tokenizer, texts, args, *, qwen, order):
    import torch

    arrays = []
    for start in range(0, len(texts), args.neural_batch_size):
        batch = texts[start:start + args.neural_batch_size]
        if qwen:
            batch = [qwen_classification_text(text) for text in batch]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           max_length=args.neural_max_length, return_tensors="pt")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.inference_mode():
            logits = model(**inputs).logits
            arrays.append(torch.sigmoid(logits.float()).cpu().numpy()[:, order])
    return np.vstack(arrays)


def predict_mlm(model, tokenizer, texts, args):
    """Cloze proxy with full verbalizers; truncation always preserves task masks."""
    import torch

    if tokenizer.mask_token_id is None:
        raise ValueError("DeBERTa cloze evaluation requires a mask token")
    targets = [tokenizer.encode(word, add_special_tokens=False) for word in VERBALIZERS]
    if any(not tokens for tokens in targets):
        raise ValueError("A verbalizer could not be tokenized")
    tokenizer.truncation_side = "left"  # The task and prediction masks are at the end.
    rows = []
    for text in texts:
        prompts = [f"Question: {text} The task is {' '.join([tokenizer.mask_token] * len(target))}."
                   for target in targets]
        scores = []
        for start in range(0, len(prompts), args.neural_batch_size):
            batch = tokenizer(prompts[start:start + args.neural_batch_size], padding=True,
                              truncation=True, max_length=args.neural_max_length, return_tensors="pt")
            batch = {key: value.to(model.device) for key, value in batch.items()}
            with torch.inference_mode():
                logits = model(**batch).logits
                for local_row, target in enumerate(targets[start:start + args.neural_batch_size]):
                    positions = torch.where(batch["input_ids"][local_row] == tokenizer.mask_token_id)[0]
                    if len(positions) < len(target):
                        raise ValueError("Prediction masks were truncated; increase --neural-max-length")
                    # Ignore any mask tokens appearing inside the user's question.
                    selected = logits[local_row, positions[-len(target):]].float().log_softmax(dim=-1)
                    scores.append(selected[torch.arange(len(target), device=model.device), target].mean())
        rows.append(torch.softmax(torch.stack(scores), dim=0).cpu().numpy())
    return np.vstack(rows)


def predict_qwen_generation(model, tokenizer, texts, args, raw_file):
    import torch

    predictions = []
    # Preserve every raw response, including the response that fails parsing.
    with raw_file.open("w", encoding="utf-8") as stream:
        for row, question in enumerate(texts):
            prompt = tokenizer.apply_chat_template(qwen_messages(question), tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=False)
            if inputs["input_ids"].shape[1] > args.qwen_max_prompt_length:
                raise ValueError(f"Qwen row {row}: prompt exceeds --qwen-max-prompt-length; increase the limit")
            inputs = {key: value.to(model.device) for key, value in inputs.items()}
            with torch.inference_mode():
                generated = model.generate(**inputs, do_sample=False, max_new_tokens=args.qwen_max_new_tokens,
                                           pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            raw = tokenizer.decode(generated[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            stream.write(json.dumps({"source_row_in_split": row, "question": question, "response": raw}) + "\n")
            stream.flush()
            predictions.append(parse_qwen_labels(raw))
    return np.vstack(predictions)


def score_neural(name, texts_by_split, args, output):
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoTokenizer

    if name not in NEURAL_CLASSIFIERS:
        raise ValueError(f"Unknown neural classifier: {name}")
    # CHANGE 3b: register ModernBERT base + checkpoint sources.
    source = {
        "deberta": args.deberta_model,
        "modernbert": args.modernbert_model,
        "qwen": args.qwen_model,
        "deberta_finetuned": args.deberta_checkpoint,
        "modernbert_finetuned": args.modernbert_checkpoint,
        "qwen_finetuned": args.qwen_checkpoint,
    }[name]
    if name.endswith("_finetuned") and not Path(source).is_dir():
        raise FileNotFoundError(f"{name} checkpoint not found: {source}")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.neural_dtype]
    common = {"local_files_only": args.local_files_only,
              "cache_dir": str(args.cache_dir) if args.cache_dir else None}
    model_kwargs = {**common, "dtype": dtype}
    model = base_model = tokenizer = None
    started = time.perf_counter()
    try:
        metadata = {**checkpoint_metadata(source), "model_id": str(source), "device": device,
                    "dtype": args.neural_dtype, "external_embeddings_used": False,
                    "max_length": args.neural_max_length, "batch_size": args.neural_batch_size}
        is_adapter = name == "qwen_finetuned" and (Path(source) / "adapter_config.json").is_file()
        tokenizer_source = source
        if is_adapter and not (Path(source) / "tokenizer_config.json").is_file():
            adapter_config = json.loads((Path(source) / "adapter_config.json").read_text())
            tokenizer_source = adapter_config.get("base_model_name_or_path")
            if not tokenizer_source:
                raise ValueError("Adapter does not identify its base model for tokenizer loading")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **common)
        if name.startswith("qwen"):
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
        if is_adapter:
            from peft import PeftConfig, PeftModel
            from peft.utils.save_and_load import load_peft_weights

            config = PeftConfig.from_pretrained(str(source), **common)
            if config.task_type != "SEQ_CLS":
                raise ValueError("Qwen adapter must be a SEQ_CLS adapter, not a causal-language-model adapter")
            base_source = config.base_model_name_or_path
            if not base_source:
                raise ValueError("Adapter does not identify its base model")
            # A classification head randomly initialized on the base must be restored by the adapter.
            weights = load_peft_weights(str(source), device="cpu", **common)
            head_keys = [key for key in weights if re.search(r"(?:^|\.)score\.(?:modules_to_save\.(?:default\.)?)?weight$", key)]
            if not head_keys or any(weights[key].shape[0] != len(LABEL_COLS) for key in head_keys):
                raise ValueError("Qwen adapter is missing its trained 12-label score.weight classification head")
            del weights
            # CHANGE 3c: fall back to torch_dtype= on transformers that predate `dtype=`.
            try:
                base_model, info = AutoModelForSequenceClassification.from_pretrained(
                    base_source, num_labels=len(LABEL_COLS), problem_type="multi_label_classification",
                    output_loading_info=True, **model_kwargs,
                )
            except TypeError:
                model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
                base_model, info = AutoModelForSequenceClassification.from_pretrained(
                    base_source, num_labels=len(LABEL_COLS), problem_type="multi_label_classification",
                    output_loading_info=True, **model_kwargs,
                )
            ensure_loaded_weights(info, allowed_missing=("score.weight",))
            model = PeftModel.from_pretrained(base_model, str(source), **common)
            metadata.update(base_model=base_source, adapter=True)
        else:
            # CHANGE 3d: route via MLM_CLASSIFIERS instead of `name == "deberta"`.
            if name in MLM_CLASSIFIERS:
                cls = AutoModelForMaskedLM
            elif name == "qwen":
                cls = AutoModelForCausalLM
            else:
                cls = AutoModelForSequenceClassification
            # CHANGE 3c: same dtype fallback as above.
            try:
                model, info = cls.from_pretrained(source, output_loading_info=True, **model_kwargs)
            except TypeError:
                model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
                model, info = cls.from_pretrained(source, output_loading_info=True, **model_kwargs)
            ensure_loaded_weights(info)
        if name.startswith("qwen"):
            model.config.pad_token_id = tokenizer.pad_token_id
        model.to(device).eval()
        metadata["revision"] = getattr(model.config, "_commit_hash", None)
        # CHANGE 3d: use MLM_CLASSIFIERS so ModernBERT inherits the cloze metadata.
        if name.endswith("_finetuned"):
            order = checkpoint_label_order(model.config, args.checkpoint_labels)
            metadata.update(label_order=LABEL_COLS.copy(),
                            output_column_order=order, score_type="sigmoid_probability", threshold_policy="dev_f1")
        else:
            order = None
            is_mlm = name in MLM_CLASSIFIERS
            metadata.update(
                score_type="cloze_candidate_probability" if is_mlm else "generated_binary_label",
                fixed_threshold=0.15 if is_mlm else 0.5,
                threshold_policy="fixed_zero_shot")
        if name in MLM_CLASSIFIERS:
            metadata["verbalizers"] = dict(zip(LABEL_COLS, VERBALIZERS))
            metadata["truncation_side"] = "left"
        if name == "qwen":
            metadata.update(system_prompt=SYSTEM_PROMPT, label_definitions=LABEL_DEFINITIONS,
                            max_prompt_length=args.qwen_max_prompt_length, max_new_tokens=args.qwen_max_new_tokens)
        if name == "qwen_finetuned":
            metadata["text_template"] = qwen_classification_text("{question}")
        timing = {"load_seconds": time.perf_counter() - started}
        scores = {}
        for split, texts in texts_by_split.items():
            started = time.perf_counter()
            print(f"Scoring {name}: {split} ({len(texts)} rows)", flush=True)
            # CHANGE 3d: route via MLM_CLASSIFIERS so ModernBERT scores through predict_mlm.
            if name in MLM_CLASSIFIERS:
                scores[split] = predict_mlm(model, tokenizer, texts, args)
            elif name == "qwen":
                scores[split] = predict_qwen_generation(model, tokenizer, texts, args, output / f"{split}_raw.jsonl")
            else:
                scores[split] = predict_classification(model, tokenizer, texts, args,
                                                       qwen=name.startswith("qwen"), order=order)
            timing[f"score_{split}_seconds"] = time.perf_counter() - started
        return scores, metadata, timing
    finally:
        del model, base_model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
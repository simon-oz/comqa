"""Compare embeddings, text-only neural baselines, score fusion, and ensembles.

Run from the repository root: python src/run_embedding_experiments.py --help
"""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import gc
import hashlib
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import sys
import time
import traceback

from joblib import dump
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from embedding_experiment_utils import (
    BASELINE_MODEL, LABEL_COLS, MODEL_PROMPTS, TEXT_COL, SafeOneVsRestClassifier,
    aggregate_classification_metrics, evaluate, load_dataset, optimize_thresholds, save_predictions, split_indices,
    write_json,
)
from neural_experiment_models import NEURAL_CLASSIFIERS, score_neural
from score_fusion import RankScoreFusion

CLASSIFIERS = ("logreg", "svm", *NEURAL_CLASSIFIERS, "ensemble")


class Tee:
    def __init__(self, original, log):
        self.original, self.log = original, log

    def write(self, text):
        self.original.write(text)
        self.log.write(text)
        self.log.flush()
        return len(text)

    def flush(self):
        self.original.flush()
        self.log.flush()


def parse_args(argv=None, *, default_classifiers=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, help="JSONL training data (also accepts .txt)")
    parser.add_argument("--test", type=Path, default=Path("data/test_dataset.jsonl"))
    parser.add_argument("--dev", type=Path, help="Explicit held-out dev JSONL; required for existing fine-tuned checkpoints")
    parser.add_argument("--models", nargs="+", choices=list(MODEL_PROMPTS), default=list(MODEL_PROMPTS))
    parser.add_argument("--classifiers", nargs="+", choices=[*CLASSIFIERS, "all"],
                        default=default_classifiers or ["logreg", "svm"])
    parser.add_argument("--deberta-model", default="microsoft/deberta-v3-large")
    parser.add_argument("--qwen-model", default="Qwen/Qwen2.5-14B-Instruct")
    parser.add_argument("--deberta-checkpoint", type=Path, default=Path("deberta_large_multilabel/final"))
    parser.add_argument("--qwen-checkpoint", type=Path, default=Path("qwen2_5_14b_multilabel/final"),
                        help="Fine-tuned sequence-classification model or SEQ_CLS PEFT adapter")
    parser.add_argument("--checkpoint-labels", nargs="+", choices=LABEL_COLS,
                        help="Training label order for checkpoints lacking named id2label metadata")
    parser.add_argument("--neural-batch-size", type=int, default=2)
    parser.add_argument("--neural-max-length", type=int, default=512)
    parser.add_argument("--neural-dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--qwen-max-prompt-length", type=int, default=2048)
    parser.add_argument("--qwen-max-new-tokens", type=int, default=192)
    parser.add_argument("--no-fusion", action="store_true", help="Evaluate text-only baselines and ensemble without pairwise fusion")
    parser.add_argument("--fusion-source", choices=["logreg", "svm"], default="logreg")
    parser.add_argument("--fusion-embedding-weight", type=float, default=0.5,
                        help="Fixed embedding share in each neural+embedding pair (default: 0.5)")
    parser.add_argument("--ensemble-members", nargs="+", choices=list(CLASSIFIERS[:-1]),
                        help="Default: all selected classifiers, excluding derived fusions; missing members fail the ensemble")
    parser.add_argument("--ensemble-weight-trials", type=int, default=0,
                        help="0: equal weights; otherwise search this many dev-only random weight candidates")
    parser.add_argument("--output-dir", type=Path, help="New/empty run directory; default: experiments/<UTC timestamp>")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-size", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="e.g. cpu, cuda, cuda:0 (default: automatic)")
    parser.add_argument("--max-seq-length", type=int, help="Optional cap; never exceeds a model's native limit")
    parser.add_argument("--cache-dir", type=Path, help="Hugging Face model cache directory")
    parser.add_argument("--local-files-only", action="store_true", help="Use downloaded models without network access")
    parser.add_argument("--legacy-e5", action="store_true", help="Omit the E5 prefix to reproduce the old input format")
    parser.add_argument("--list-models", action="store_true", help="Print model IDs and classification prompts, then exit")
    parser.add_argument("--refresh-summary", type=Path,
                        help="Export an existing summary.json with precision/recall columns; requires --output-dir, runs no models")
    args = parser.parse_args(argv)
    if not 0 < args.dev_size < 1:
        parser.error("--dev-size must be between 0 and 1")
    if args.batch_size < 1 or (args.max_seq_length is not None and args.max_seq_length < 1):
        parser.error("--batch-size and --max-seq-length must be positive")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    if min(args.neural_batch_size, args.neural_max_length, args.qwen_max_prompt_length, args.qwen_max_new_tokens) < 1:
        parser.error("Neural batch size and token limits must be positive")
    if not 0 < args.fusion_embedding_weight < 1:
        parser.error("--fusion-embedding-weight must be strictly between 0 and 1")
    if args.ensemble_weight_trials < 0:
        parser.error("--ensemble-weight-trials cannot be negative")
    if args.checkpoint_labels and (len(args.checkpoint_labels) != len(LABEL_COLS) or set(args.checkpoint_labels) != set(LABEL_COLS)):
        parser.error("--checkpoint-labels must name each of the 12 labels exactly once")
    args.models = list(dict.fromkeys(args.models))
    if "all" in args.classifiers:
        args.classifiers = list(CLASSIFIERS)
    args.classifiers = list(dict.fromkeys(args.classifiers))
    if args.ensemble_members:
        if "ensemble" not in args.classifiers:
            parser.error("--ensemble-members requires --classifiers ensemble (or all)")
        args.classifiers = list(dict.fromkeys([*args.classifiers, *args.ensemble_members]))
    if not args.no_fusion and set(args.classifiers) & set(NEURAL_CLASSIFIERS):
        args.classifiers = list(dict.fromkeys([*args.classifiers, args.fusion_source]))
    if "ensemble" in args.classifiers:
        args.ensemble_members = list(dict.fromkeys(args.ensemble_members or [name for name in args.classifiers if name != "ensemble"]))
        if len(args.ensemble_members) < 2:
            parser.error("Ensemble needs at least two members; use --classifiers all or --ensemble-members")
    return args


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_versions():
    versions = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scikit-learn", "joblib", "sentence-transformers", "transformers", "torch", "sentencepiece", "peft"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def model_directory(model_id):
    return model_id.replace("/", "__")


def encode_datasets(model_id, texts_by_split, args):
    # Keep heavy optional imports out of CLI help and offline classifier tests.
    import torch
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(args.seed)
    prompt = MODEL_PROMPTS[model_id]
    if args.legacy_e5 and model_id.startswith("intfloat/e5-"):
        prompt = ""
    started = time.perf_counter()
    embedder = None
    try:
        embedder = SentenceTransformer(
            model_id, device=args.device,
            cache_folder=str(args.cache_dir) if args.cache_dir else None,
            local_files_only=args.local_files_only,
        )
        # EmbeddingGemma does not support float16 activations. Use float32 for all models.
        embedder.float()
        if args.max_seq_length is not None:
            embedder.max_seq_length = min(args.max_seq_length, embedder.max_seq_length)
        first_module = embedder[0]
        config = getattr(getattr(first_module, "auto_model", None), "config", None)
        metadata = {
            "model_id": model_id, "prompt": prompt,
            "normalize_embeddings": True, "dtype": "float32",
            "max_seq_length": int(embedder.max_seq_length),
            "revision": getattr(config, "_commit_hash", None),
            "device": str(embedder.device), "batch_size": args.batch_size,
        }
        timings = {"load_seconds": time.perf_counter() - started}
        embeddings = {}
        for split, texts in texts_by_split.items():
            print(f"Encoding {model_id}: {split} ({len(texts)} rows)", flush=True)
            started = time.perf_counter()
            # An explicit empty prompt disables checkpoint defaults for unprompted models.
            array = np.asarray(embedder.encode(
                texts, prompt=prompt, batch_size=args.batch_size,
                normalize_embeddings=True, convert_to_numpy=True,
                show_progress_bar=False,
            ), dtype=np.float32)
            if array.ndim != 2 or array.shape[0] != len(texts) or not np.isfinite(array).all():
                raise ValueError(f"Invalid {split} embeddings from {model_id}")
            embeddings[split] = array
            timings[f"encode_{split}_seconds"] = time.perf_counter() - started
        dimensions = {array.shape[1] for array in embeddings.values()}
        if len(dimensions) != 1 or 0 in dimensions:
            raise ValueError("Embedding dimensions must be nonzero and agree across splits")
        metadata["dimension"] = dimensions.pop()
        return embeddings, metadata, timings
    finally:
        del embedder
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def make_classifier(name, seed):
    if name == "logreg":
        estimator = LogisticRegression(
            C=0.3, class_weight="balanced", solver="liblinear",
            max_iter=20000, random_state=seed,
        )
    else:
        estimator = LinearSVC(C=3.0, class_weight="balanced", max_iter=20000, random_state=seed)
    return SafeOneVsRestClassifier(estimator)


def fit_and_save(name, model_id, X, scaler, frames, rows, has_test_labels,
                 embedding_metadata, embedding_timing, args, output):
    output.mkdir()
    model = make_classifier(name, args.seed)
    prefer_proba = name == "logreg"
    started = time.perf_counter()
    model.fit(X["train"], frames["train"][LABEL_COLS].to_numpy())
    timing = dict(embedding_timing, train_seconds=time.perf_counter() - started)
    started = time.perf_counter()
    dev_scores = model.scores(X["dev"], prefer_proba=prefer_proba)
    y_dev = frames["dev"][LABEL_COLS].to_numpy()
    thresholds = optimize_thresholds(y_dev, dev_scores)
    dev_metrics = evaluate(y_dev, dev_scores, thresholds)
    timing["dev_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    test_scores = model.scores(X["test"], prefer_proba=prefer_proba)
    test_metrics = evaluate(frames["test"][LABEL_COLS].to_numpy(), test_scores, thresholds) if has_test_labels else None
    timing["test_seconds"] = time.perf_counter() - started
    result = {
        "status": "completed", "model_id": model_id, "classifier": name,
        "evaluation_kind": "embedding_classifier", "embedding_dependent": True,
        "embedding": embedding_metadata, "classifier_params": model.estimator.get_params(),
        "score_type": "probability" if prefer_proba else "decision_function",
        "thresholds": dict(zip(LABEL_COLS, thresholds.tolist())),
        "timing": timing, "dev": dev_metrics, "test": test_metrics,
    }
    for split, scores in (("dev", dev_scores), ("test", test_scores)):
        save_predictions(frames[split], rows[split], scores, thresholds,
                         output / f"{split}_predictions.csv",
                         has_labels=split == "dev" or has_test_labels)
    np.save(output / "thresholds.npy", thresholds)
    np.savez_compressed(output / "scores.npz", dev=dev_scores, test=test_scores)
    # A single bundle keeps the model, scaler and preprocessing contract together.
    dump({"classifier": model, "scaler": scaler, "thresholds": thresholds,
          "label_cols": LABEL_COLS, "embedding": embedding_metadata,
          "prefer_proba": prefer_proba}, output / "model.joblib")
    write_json(output / "metrics.json", result)
    print(f"{model_id} / {name}: dev F1 micro={dev_metrics['f1_micro']:.4f}"
          + (f", test F1 micro={test_metrics['f1_micro']:.4f}" if test_metrics else ", test unlabeled"))
    return result


def save_score_evaluation(name, model_id, scores, frames, rows, has_test_labels,
                          metadata, timing, output, *, thresholds=None):
    output.mkdir(parents=True, exist_ok=True)
    for split in ("dev", "test"):
        if scores[split].shape != (len(frames[split]), len(LABEL_COLS)) or not np.isfinite(scores[split]).all():
            raise ValueError(f"Invalid/unaligned {name} {split} score matrix")
    if thresholds is None:
        thresholds = optimize_thresholds(frames["dev"][LABEL_COLS].to_numpy(), scores["dev"])
    thresholds = np.asarray(thresholds, dtype=float)
    result = {"status": "completed", "model_id": model_id, "classifier": name,
              **metadata, "timing": timing, "thresholds": dict(zip(LABEL_COLS, thresholds.tolist()))}
    for split in ("dev", "test"):
        labeled = split == "dev" or has_test_labels
        result[split] = evaluate(frames[split][LABEL_COLS].to_numpy(), scores[split], thresholds) if labeled else None
        save_predictions(frames[split], rows[split], scores[split], thresholds,
                         output / f"{split}_predictions.csv", has_labels=labeled)
    np.save(output / "thresholds.npy", thresholds)
    np.savez_compressed(output / "scores.npz", **scores)
    write_json(output / "metrics.json", result)
    return result


def evaluate_fusion(name, model_id, members, score_store, frames, rows, has_test_labels,
                    args, output, *, embedding=None, weights=None, baseline=None):
    missing = [member for member in members if member not in score_store]
    if missing:
        raise ValueError(f"Cannot evaluate {name}; required members failed: {', '.join(missing)}")
    started = time.perf_counter()
    dev = {member: score_store[member]["dev"] for member in members}
    trials = args.ensemble_weight_trials if name.startswith("ensemble") else 0
    fusion = RankScoreFusion(weights=weights, weight_trials=trials, seed=args.seed)
    fusion.fit(dev, frames["dev"][LABEL_COLS].to_numpy())
    scores = {split: fusion.transform({member: score_store[member][split] for member in members})
              for split in ("dev", "test")}
    metadata = {"evaluation_kind": "ensemble" if name.startswith("ensemble") else "neural_embedding_fusion",
                "embedding_dependent": embedding is not None, "members": members,
                "weights": fusion.weights_, "score_type": "dev_percentile_rank_fusion",
                "threshold_policy": "dev_f1", "weight_policy": "dev_ap_macro_search" if trials else "fixed",
                "weight_trials": trials}
    if embedding is not None:
        metadata["embedding"] = embedding
    result = save_score_evaluation(name, model_id, scores, frames, rows, has_test_labels,
                                   metadata, {"fusion_seconds": time.perf_counter() - started}, output)
    if baseline is not None:
        result["baseline_classifier"] = baseline["classifier"]
        for split in ("dev", "test"):
            if result[split] is not None and baseline[split] is not None:
                for metric in ("f1_micro", "f1_macro", "exact_match_ratio", "hamming_loss"):
                    result[f"delta_{split}_{metric}"] = result[split][metric] - baseline[split][metric]
        write_json(output / "metrics.json", result)
    dump({"fusion": fusion, "thresholds": np.load(output / "thresholds.npy"),
          "label_cols": LABEL_COLS, "members": members}, output / "fusion.joblib")
    return result


def save_summary(output, results):
    write_json(output / "summary.json", results)
    flat = []
    for result in results:
        row = {key: result.get(key) for key in ("model_id", "classifier", "status", "error",
               "evaluation_kind", "embedding_dependent", "baseline_classifier", "score_type")}
        row.update({key: value for key, value in result.items() if key.startswith("delta_")})
        row.update({f"embedding_{key}": value for key, value in result.get("embedding", {}).items()})
        row.update(result.get("timing", {}))
        for split in ("dev", "test"):
            row.update({f"{split}_{key}": value for key, value in (result.get(split) or {}).items()
                        if key != "classification_report"})
        flat.append(row)
    temporary = output / "summary.csv.tmp"
    pd.DataFrame(flat).to_csv(temporary, index=False)
    temporary.replace(output / "summary.csv")


def refresh_summary(source, output):
    """Recover aggregate metrics from saved reports without fitting or inference."""
    if output is None:
        raise ValueError("--refresh-summary requires a new/empty --output-dir")
    results = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(results, list) or not all(isinstance(result, dict) for result in results):
        raise ValueError("--refresh-summary expects the result list from summary.json")
    for result in results:
        for split in ("dev", "test"):
            metrics = result.get(split)
            if metrics is None:
                continue
            try:
                aggregates = aggregate_classification_metrics(metrics["classification_report"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Missing or invalid {split} classification_report for "
                    f"{result.get('model_id')} / {result.get('classifier')}; "
                    "use the original full summary.json, not the CSV"
                ) from exc
            for key, value in aggregates.items():
                metrics.setdefault(key, value)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output directory must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    save_summary(output, results)
    print(f"Results: {output / 'summary.csv'} (regenerated from {source}; no models run)")
    return 0


def run(args, *, encoder=None, neural_scorer=None):
    """Scorer hooks support offline pipeline tests; the CLI uses real checkpoints."""
    if args.train is None:
        args.train = next((Path(path) for path in (
            "data/train_dataset.txt", "train_dataset.txt", "data/train_dataset.jsonl", "train_dataset.jsonl",
        ) if Path(path).is_file()), None)
    if args.train is None:
        raise FileNotFoundError("Training data not found. Pass --train /path/to/train_dataset.jsonl")
    train_df, _ = load_dataset(args.train, require_labels=True)
    test_df, has_test_labels = load_dataset(args.test, require_labels=False)
    if args.dev is None:
        if any(name.endswith("_finetuned") for name in args.classifiers):
            raise ValueError("Existing fine-tuned checkpoints require --dev with data held out from their training. "
                             "Use --train containing only training rows; a new random split cannot prove a checkpoint held dev out.")
        train_rows, dev_rows, strategy = split_indices(train_df, args.dev_size, args.seed)
        dev_df = train_df.iloc[dev_rows]
    else:
        dev_df, _ = load_dataset(args.dev, require_labels=True)
        if set(dev_df[TEXT_COL]) & (set(train_df[TEXT_COL]) | set(test_df[TEXT_COL])):
            raise ValueError("Explicit --dev questions must not overlap --train or --test questions")
        train_rows, dev_rows, strategy = np.arange(len(train_df)), np.arange(len(dev_df)), "explicit_dev_file"
    frames = {"train": train_df.iloc[train_rows], "dev": dev_df, "test": test_df}
    rows = {"train": train_rows, "dev": dev_rows, "test": np.arange(len(test_df))}
    texts = {split: frame[TEXT_COL].tolist() for split, frame in frames.items()}
    output = args.output_dir or Path("experiments") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be new or empty (existing results are protected): {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "output_dir": str(output), "label_cols": LABEL_COLS, "versions": package_versions(),
        "datasets": {name: {"path": str(path.resolve()), "sha256": fingerprint(path)}
                     for name, path in (("train", args.train), ("test", args.test))},
        "split_strategy": strategy, "split_rows": {key: value.tolist() for key, value in rows.items()},
        "row_counts": {key: len(frame) for key, frame in frames.items()},
        "test_has_labels": has_test_labels,
        "train_test_question_overlap": len(set(train_df[TEXT_COL]) & set(test_df[TEXT_COL])),
        "train_dev_question_overlap": len(set(texts["train"]) & set(texts["dev"])),
    }
    if args.dev is not None:
        manifest["datasets"]["dev"] = {"path": str(args.dev.resolve()), "sha256": fingerprint(args.dev)}
    manifest["checkpoint_dev_exclusion"] = "user_supplied_held_out_dev" if args.dev else "not_applicable"
    write_json(output / "manifest.json", manifest)
    results = []
    encoder = encoder or encode_datasets
    neural_scorer = neural_scorer or score_neural
    neural_names = [name for name in args.classifiers if name in NEURAL_CLASSIFIERS]
    classical_names = [name for name in args.classifiers if name in ("logreg", "svm")]
    baseline_scores, baseline_results = {}, {}
    with (output / "run.log").open("w", encoding="utf-8") as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            print(f"Saving experiment results to {output}", flush=True)
            print(f"Split: {strategy}; sizes: {manifest['row_counts']}")
            for key in ("train_test_question_overlap", "train_dev_question_overlap"):
                if manifest[key]:
                    print(f"WARNING: {key}={manifest[key]}; duplicate questions may inflate metrics.")
            def record_failure(name, model_id, exc, folder, kind):
                traceback.print_exc()
                folder.mkdir(parents=True, exist_ok=True)
                result = {"model_id": model_id, "classifier": name, "status": "failed",
                          "evaluation_kind": kind, "error": f"{type(exc).__name__}: {exc}"}
                write_json(folder / "error.json", result)
                results.append(result)
                save_summary(output, results)

            for name in neural_names:
                folder = output / "baselines" / name
                folder.mkdir(parents=True)
                try:
                    # No train or test labels reach neural inference.
                    scores, neural_metadata, timing = neural_scorer(
                        name, {split: texts[split] for split in ("dev", "test")}, args, folder,
                    )
                    fixed = neural_metadata.get("fixed_threshold")
                    result = save_score_evaluation(
                        name, neural_metadata["model_id"], scores, frames, rows, has_test_labels,
                        {"evaluation_kind": "text_only_baseline", "embedding_dependent": False,
                         "neural": neural_metadata, "score_type": neural_metadata["score_type"]}, timing, folder,
                        thresholds=np.full(len(LABEL_COLS), fixed) if fixed is not None else None,
                    )
                    baseline_scores[name], baseline_results[name] = scores, result
                    results.append(result)
                    save_summary(output, results)
                except Exception as exc:
                    record_failure(name, None, exc, folder, "text_only_baseline")

            text_ensemble = None
            text_members = [name for name in (args.ensemble_members or []) if name in NEURAL_CLASSIFIERS]
            if "ensemble" in args.classifiers and len(text_members) >= 2:
                try:
                    text_ensemble = evaluate_fusion(
                        "ensemble_text_only", "text_only", text_members, baseline_scores, frames, rows,
                        has_test_labels, args, output / "baselines" / "ensemble_text_only",
                    )
                    results.append(text_ensemble)
                    save_summary(output, results)
                except Exception as exc:
                    record_failure("ensemble_text_only", "text_only", exc,
                                   output / "baselines" / "ensemble_text_only", "ensemble")
            # With no embedding consumers, shared neural baselines are the whole run.
            for model_id in args.models if classical_names else []:
                model_output = output / model_directory(model_id)
                model_output.mkdir()
                score_store = dict(baseline_scores)
                embedding_metadata = None
                X = scaler = None
                try:
                    X, embedding_metadata, timing = encoder(model_id, texts, args)
                    started = time.perf_counter()
                    scaler = StandardScaler()
                    X["train"] = scaler.fit_transform(X["train"]).astype(np.float32)
                    for split in ("dev", "test"):
                        X[split] = scaler.transform(X[split]).astype(np.float32)
                    timing["scale_seconds"] = time.perf_counter() - started
                    write_json(model_output / "embedding.json", {**embedding_metadata, "timing": timing})
                except Exception as exc:
                    embedding_metadata = None
                    traceback.print_exc()
                    error = f"{type(exc).__name__}: {exc}"
                    write_json(model_output / "error.json", {"model_id": model_id, "error": error})
                    for name in classical_names:
                        results.append({"model_id": model_id, "classifier": name, "status": "failed",
                                        "evaluation_kind": "embedding_classifier", "embedding_dependent": True,
                                        "error": error})
                    save_summary(output, results)
                for name in classical_names if embedding_metadata is not None else []:
                    try:
                        result = fit_and_save(name, model_id, X, scaler, frames, rows, has_test_labels,
                                              embedding_metadata, timing, args, model_output / name)
                        with np.load(model_output / name / "scores.npz") as saved:
                            score_store[name] = {split: saved[split] for split in ("dev", "test")}
                    except Exception as exc:
                        traceback.print_exc()
                        result = {"model_id": model_id, "classifier": name, "status": "failed",
                                  "error": f"{type(exc).__name__}: {exc}"}
                        write_json(model_output / f"{name}_error.json", result)
                    results.append(result)
                    save_summary(output, results)
                if not args.no_fusion:
                    for name in neural_names:
                        fusion_name = f"{name}+{args.fusion_source}"
                        try:
                            weights = {name: 1 - args.fusion_embedding_weight,
                                       args.fusion_source: args.fusion_embedding_weight}
                            result = evaluate_fusion(
                                fusion_name, model_id, [name, args.fusion_source], score_store, frames, rows,
                                has_test_labels, args, model_output / fusion_name, embedding=embedding_metadata,
                                weights=weights, baseline=baseline_results.get(name),
                            )
                            results.append(result)
                            save_summary(output, results)
                        except Exception as exc:
                            record_failure(fusion_name, model_id, exc, model_output / fusion_name, "neural_embedding_fusion")
                if "ensemble" in args.classifiers and any(name in classical_names for name in args.ensemble_members):
                    try:
                        result = evaluate_fusion(
                            "ensemble", model_id, args.ensemble_members, score_store, frames, rows,
                            has_test_labels, args, model_output / "ensemble", embedding=embedding_metadata,
                            baseline=text_ensemble,
                        )
                        results.append(result)
                        save_summary(output, results)
                    except Exception as exc:
                        record_failure("ensemble", model_id, exc, model_output / "ensemble", "ensemble")
                del X, scaler
                gc.collect()
    failed = any(result["status"] != "completed" for result in results)
    manifest.update(status="completed_with_failures" if failed else "completed",
                    finished_at=datetime.now(timezone.utc).isoformat())
    write_json(output / "manifest.json", manifest)
    print(f"Results: {output / 'summary.csv'}")
    return 1 if failed else 0


def main(argv=None, *, default_classifiers=None):
    args = parse_args(argv, default_classifiers=default_classifiers)
    if args.list_models:
        for model_id, prompt in MODEL_PROMPTS.items():
            print(f"{model_id}\tprompt={prompt!r}" + (" (baseline)" if model_id == BASELINE_MODEL else ""))
        return 0
    try:
        if args.refresh_summary is not None:
            return refresh_summary(args.refresh_summary, args.output_dir)
        return run(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Experiment setup failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

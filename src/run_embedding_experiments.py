"""Compare frozen sentence embeddings with the existing logistic/SVM protocol.

Run from the repository root: python src/run_embedding_experiments.py --help
"""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import gc
import hashlib
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
    evaluate, load_dataset, optimize_thresholds, save_predictions, split_indices,
    write_json,
)


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
    parser.add_argument("--models", nargs="+", choices=list(MODEL_PROMPTS), default=list(MODEL_PROMPTS))
    parser.add_argument("--classifiers", nargs="+", choices=["logreg", "svm"],
                        default=default_classifiers or ["logreg", "svm"])
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
    args = parser.parse_args(argv)
    if not 0 < args.dev_size < 1:
        parser.error("--dev-size must be between 0 and 1")
    if args.batch_size < 1 or (args.max_seq_length is not None and args.max_seq_length < 1):
        parser.error("--batch-size and --max-seq-length must be positive")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 2**32 - 1")
    args.models = list(dict.fromkeys(args.models))
    args.classifiers = list(dict.fromkeys(args.classifiers))
    return args


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def package_versions():
    versions = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scikit-learn", "joblib", "sentence-transformers", "transformers", "torch", "sentencepiece"):
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
    # A single bundle keeps the model, scaler and preprocessing contract together.
    dump({"classifier": model, "scaler": scaler, "thresholds": thresholds,
          "label_cols": LABEL_COLS, "embedding": embedding_metadata,
          "prefer_proba": prefer_proba}, output / "model.joblib")
    write_json(output / "metrics.json", result)
    print(f"{model_id} / {name}: dev F1 micro={dev_metrics['f1_micro']:.4f}"
          + (f", test F1 micro={test_metrics['f1_micro']:.4f}" if test_metrics else ", test unlabeled"))
    return result


def save_summary(output, results):
    write_json(output / "summary.json", results)
    flat = []
    for result in results:
        row = {key: result.get(key) for key in ("model_id", "classifier", "status", "error")}
        row.update({f"embedding_{key}": value for key, value in result.get("embedding", {}).items()})
        row.update(result.get("timing", {}))
        for split in ("dev", "test"):
            row.update({f"{split}_{key}": value for key, value in (result.get(split) or {}).items()
                        if key != "classification_report"})
        flat.append(row)
    temporary = output / "summary.csv.tmp"
    pd.DataFrame(flat).to_csv(temporary, index=False)
    temporary.replace(output / "summary.csv")


def run(args, *, encoder=None):
    """The encoder hook supports offline pipeline tests; the CLI always uses real models."""
    if args.train is None:
        args.train = next((Path(path) for path in (
            "data/train_dataset.txt", "train_dataset.txt", "data/train_dataset.jsonl", "train_dataset.jsonl",
        ) if Path(path).is_file()), None)
    if args.train is None:
        raise FileNotFoundError("Training data not found. Pass --train /path/to/train_dataset.jsonl")
    train_df, _ = load_dataset(args.train, require_labels=True)
    test_df, has_test_labels = load_dataset(args.test, require_labels=False)
    train_rows, dev_rows, strategy = split_indices(train_df, args.dev_size, args.seed)
    frames = {"train": train_df.iloc[train_rows], "dev": train_df.iloc[dev_rows], "test": test_df}
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
    write_json(output / "manifest.json", manifest)
    results = []
    encoder = encoder or encode_datasets
    with (output / "run.log").open("w", encoding="utf-8") as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            print(f"Saving experiment results to {output}", flush=True)
            print(f"Split: {strategy}; sizes: {manifest['row_counts']}")
            for key in ("train_test_question_overlap", "train_dev_question_overlap"):
                if manifest[key]:
                    print(f"WARNING: {key}={manifest[key]}; duplicate questions may inflate metrics.")
            for model_id in args.models:
                model_output = output / model_directory(model_id)
                model_output.mkdir()
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
                    traceback.print_exc()
                    error = f"{type(exc).__name__}: {exc}"
                    write_json(model_output / "error.json", {"model_id": model_id, "error": error})
                    for name in args.classifiers:
                        results.append({"model_id": model_id, "classifier": name, "status": "failed", "error": error})
                    save_summary(output, results)
                    continue
                for name in args.classifiers:
                    try:
                        result = fit_and_save(name, model_id, X, scaler, frames, rows, has_test_labels,
                                              embedding_metadata, timing, args, model_output / name)
                    except Exception as exc:
                        traceback.print_exc()
                        result = {"model_id": model_id, "classifier": name, "status": "failed",
                                  "error": f"{type(exc).__name__}: {exc}"}
                        write_json(model_output / f"{name}_error.json", result)
                    results.append(result)
                    save_summary(output, results)
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
        return run(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Experiment setup failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

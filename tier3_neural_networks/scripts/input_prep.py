#!/usr/bin/env python3

import argparse
import random
import pandas as pd

from shared.config_utils import get_resolved_paths, load_config
from shared.json_and_csv_utils import read_index_file, write_index_file, write_json


def parse_args() -> argparse.Namespace:
    # returns path to config file
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",type=str,default="config.yaml",help="path to config.yaml")
    return parser.parse_args()


def load_manifest(manifest_path):
    # returns a manifest csv as a df 

    manifest_df = pd.read_csv(manifest_path)
    required_cols = {"meta_index", "render_directory"}
    missing = required_cols - set(manifest_df.columns)
    if missing:
        raise ValueError("Manifest must contain columns meta_index and render_directory")
    manifest_df = manifest_df.copy()
    manifest_df["meta_index"] = manifest_df["meta_index"].astype(int)
    manifest_df["render_directory"] = manifest_df["render_directory"].astype(str)

    return manifest_df


def validate_splits(train_idx, val_idx, test_idx, valid_meta_indices):
    # determines if split txt files are valid if provided

    train_set = set(train_idx)
    val_set = set(val_idx)
    test_set = set(test_idx)

    if not train_idx or not val_idx or not test_idx:
        return False, "one or more split files are empty"
    if len(train_idx) != len(train_set):
        return False, "train split contains duplicate meta_index values"
    if len(val_idx) != len(val_set):
        return False, "validation split contains duplicate meta_index values"
    if len(test_idx) != len(test_set):
        return False, "test split contains duplicate meta_index values"

    if train_set & val_set:
        return False, "train and validation splits overlap"
    if train_set & test_set:
        return False, "train and test splits overlap"
    if val_set & test_set:
        return False, "validation and test splits overlap"

    for split_name, split_set in (
        ("train", train_set),
        ("validation", val_set),
        ("test", test_set)
        ):
        invalid_indices = sorted(split_set - valid_meta_indices)
        if invalid_indices:
            return (
                False,
                f"{split_name} split contains meta_index values not present in "
                f"the manifest, e.g. {invalid_indices[:10]}"
                )

    return True, "valid"


def splits_are_valid(train_idx, val_idx, test_idx, valid_meta_indices):
    is_valid, _ = validate_splits(train_idx, val_idx, test_idx, valid_meta_indices)
    return is_valid


def make_stratified_random_splits(manifest_df, seed, train_frac=0.70, val_frac=0.15, test_frac=0.15):
    # split generator if no valid split files provided
    from sklearn.model_selection import train_test_split
    
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-9:
        raise ValueError("split fractions must sum to 1.0")

    # load
    df = manifest_df[["meta_index", "render_directory"]].copy()

    # first split is train vs temp
    train_df, temp_df = train_test_split(
        df,
        test_size = (1.0 - train_frac),
        random_state = seed,
        stratify = df["render_directory"]
        )

    # second split is val vs test (within temp)
    val_relative = val_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        temp_df,
        test_size = (1.0 - val_relative),
        random_state = seed,
        stratify = temp_df["render_directory"]
        )

    train_idx = train_df["meta_index"].astype(int).tolist()
    val_idx = val_df["meta_index"].astype(int).tolist()
    test_idx = test_df["meta_index"].astype(int).tolist()

    # return train/val/test indices
    return train_idx, val_idx, test_idx


def read_and_validate_split_triplet(train_path, val_path, test_path, valid_meta_indices):
    train_idx = read_index_file(train_path)
    val_idx = read_index_file(val_path)
    test_idx = read_index_file(test_path)
    splits_valid, validation_message = validate_splits(
        train_idx,
        val_idx,
        test_idx,
        valid_meta_indices
        )
    if not splits_valid:
        raise ValueError(validation_message)
    return train_idx, val_idx, test_idx


def shuffle_split_indices(indices, seed):
    shuffled_indices = sorted(int(index) for index in indices)  # sort for idempotent shuffling with multiple runs
    random.Random(seed).shuffle(shuffled_indices)
    return shuffled_indices


def main():
    args = parse_args()
    cfg = load_config(args.config)
    resolved_paths = get_resolved_paths(cfg, args.config)

    run_name = resolved_paths["run_name"]
    run_dir = resolved_paths["run_dir"]
    manifest_path = resolved_paths["manifest"]
    train_path = resolved_paths["train_indices"]
    val_path = resolved_paths["val_indices"]
    test_path = resolved_paths["test_indices"]
    source_train_path = resolved_paths["source_train_indices"]
    source_val_path = resolved_paths["source_val_indices"]
    source_test_path = resolved_paths["source_test_indices"]
    artifacts_dir = resolved_paths["artifacts_dir"]

    manifest_df = load_manifest(manifest_path)
    valid_meta_indices = set(manifest_df["meta_index"].tolist())

    source = None
    candidate_failures = {}
    fallback_reason = None

    split_candidates = [
        ("provided_run_splits", train_path, val_path, test_path),
        ("provided_split_source", source_train_path, source_val_path, source_test_path),
    ]

    for candidate_name, candidate_train, candidate_val, candidate_test in split_candidates:
        try:
            train_idx, val_idx, test_idx = read_and_validate_split_triplet(
                candidate_train,
                candidate_val,
                candidate_test,
                valid_meta_indices,
            )
            source = candidate_name
            break
        except Exception as exc:
            candidate_failures[candidate_name] = f"{type(exc).__name__}: {exc}"

    if source is None:
        source = "random_fallback_stratified"
        fallback_reason = "; ".join(
            f"{name}: {reason}"
            for name, reason in candidate_failures.items()
        )
        train_idx, val_idx, test_idx = make_stratified_random_splits(
            manifest_df=manifest_df,
            seed=int(cfg["seed"])
            )

    train_idx = shuffle_split_indices(train_idx, int(cfg["seed"]) + 101)
    val_idx = shuffle_split_indices(val_idx, int(cfg["seed"]) + 102)
    test_idx = shuffle_split_indices(test_idx, int(cfg["seed"]) + 103)

    write_index_file(train_idx, train_path)
    write_index_file(val_idx, val_path)
    write_index_file(test_idx, test_path)

    summary = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "source": source,
        "seed": int(cfg["seed"]),
        "split_source_paths": {
            "train": str(source_train_path),
            "val": str(source_val_path),
            "test": str(source_test_path)
            },
        "candidate_failures": candidate_failures,
        "counts": {
            "manifest": int(len(manifest_df)),
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx))
            },
        "split_indices_shuffled": True,
        "stratify_by": "render_directory" if source == "random_fallback_stratified" else None,
        "fallback_reason": fallback_reason
        }

    write_json(summary, artifacts_dir / "splits" / "split_summary.json")

    print(f"Run name: {run_name}")
    print(f"Run dir: {run_dir}")
    print(f"Wrote train indices: {train_path}")
    print(f"Wrote val indices: {val_path}")
    print(f"Wrote test indices: {test_path}")
    print(f"Split source: {source}")
    if fallback_reason:
        print(f"Fallback reason: {fallback_reason}")


if __name__ == "__main__":
    main()

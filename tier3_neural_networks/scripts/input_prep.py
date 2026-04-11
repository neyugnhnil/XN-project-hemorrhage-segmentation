#!/usr/bin/env python3

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

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


def splits_are_valid(train_idx, val_idx, test_idx, valid_meta_indices):
    # determines if split txt files are valid if provided

    train_set = set(train_idx)
    val_set = set(val_idx)
    test_set = set(test_idx)

    if not train_idx or not val_idx or not test_idx:
        return False
    if len(train_idx) != len(train_set):
        return False
    if len(val_idx) != len(val_set):
        return False
    if len(test_idx) != len(test_set):
        return False

    if train_set & val_set:
        return False
    if train_set & test_set:
        return False
    if val_set & test_set:
        return False

    if not train_set.issubset(valid_meta_indices):
        return False
    if not val_set.issubset(valid_meta_indices):
        return False
    if not test_set.issubset(valid_meta_indices):
        return False

    return True


def make_stratified_random_splits(manifest_df, seed, train_frac=0.70, val_frac=0.15, test_frac=0.15):
    # split generator if no valid split files provided
    
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


def main():
    args = parse_args()
    cfg = load_config(args.config)
    resolved_paths = get_resolved_paths(cfg, args.config)

    manifest_path = resolved_paths["manifest"]
    train_path = resolved_paths["train_indices"]
    val_path = resolved_paths["val_indices"]
    test_path = resolved_paths["test_indices"]
    artifacts_dir = resolved_paths["artifacts_dir"]

    manifest_df = load_manifest(manifest_path)
    valid_meta_indices = set(manifest_df["meta_index"].tolist())

    source = "provided"

    try:
        train_idx = read_index_file(train_path)
        val_idx = read_index_file(val_path)
        test_idx = read_index_file(test_path)
        if not splits_are_valid(train_idx, val_idx, test_idx, valid_meta_indices):
            raise ValueError("provided splits are invalid")

    except Exception:
        source = "random_fallback_stratified"
        train_idx, val_idx, test_idx = make_stratified_random_splits(
            manifest_df=manifest_df,
            seed=int(cfg["seed"])
            )

    write_index_file(train_idx, train_path)
    write_index_file(val_idx, val_path)
    write_index_file(test_idx, test_path)

    summary = {
        "source": source,
        "seed": int(cfg["seed"]),
        "counts": {
            "manifest": int(len(manifest_df)),
            "train": int(len(train_idx)),
            "val": int(len(val_idx)),
            "test": int(len(test_idx))
            },
            "stratify_by": "render_directory" if source != "provided" else None
            }

    write_json(summary, artifacts_dir / "splits" / "split_summary.json")

    print(f"Wrote train indices: {train_path}")
    print(f"Wrote val indices: {val_path}")
    print(f"Wrote test indices: {test_path}")
    print(f"Split source: {source}")


if __name__ == "__main__":
    main()
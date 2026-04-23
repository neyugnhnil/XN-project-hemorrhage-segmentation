#!/usr/bin/env python3
from pathlib import Path
from typing import Any
import yaml

# read/load yaml files
def load_config(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# turn relative paths to absolute
def resolve_project_root(cfg: dict[str, Any], config_path: str | Path) -> Path:
    config_path = Path(config_path).resolve()
    config_dir = config_path.parent
    project_root_value = cfg.get("project_root", ".")
    return (config_dir / project_root_value).resolve()
def resolve_from_project_root(project_root: str | Path, relative_path: str | Path) -> Path:
    return (Path(project_root) / Path(relative_path)).resolve()


def get_run_name(cfg: dict[str, Any]) -> str:
    run_name = str(cfg.get("run_name", "default")).strip()
    if not run_name:
        raise ValueError("run_name must not be empty")
    if "/" in run_name or "\\" in run_name:
        raise ValueError("run_name must be a plain folder name, not a path")
    return run_name


def get_attentioncnn_segmenter_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg["attentioncnn"]["segmenter"]


def get_attentioncnn_classifier_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return cfg["attentioncnn"]["classifier"]


def get_attentioncnn_attention_floor(cfg: dict[str, Any]) -> float:
    attention_floor = float(cfg["attentioncnn"].get("attention_floor", 0.0))
    if attention_floor < 0.0 or attention_floor > 1.0:
        raise ValueError("attentioncnn.attention_floor must be between 0.0 and 1.0")
    return attention_floor


def get_model_batch_size(cfg: dict[str, Any], model_family: str) -> int:
    if model_family == "attentioncnn":
        return int(get_attentioncnn_classifier_config(cfg)["batch_size"])
    return int(cfg[model_family]["batch_size"])


# bundle resolved paths into dictionary for the whole pipeline
def get_resolved_paths(cfg: dict[str, Any], config_path: str | Path) -> dict[str, Path]:
    project_root = resolve_project_root(cfg, config_path)
    paths_cfg = cfg["paths"]
    run_name = get_run_name(cfg)
    runs_dir = resolve_from_project_root(project_root, paths_cfg.get("runs_dir", "runs"))
    run_dir = runs_dir / run_name
    split_source_dir = resolve_from_project_root(project_root, paths_cfg.get("split_source_dir", "splits"))
    split_dir = run_dir / "splits"
    artifacts_dir = run_dir / "artifacts"
    results_dir = run_dir / "results"

    resolved = {
        "project_root": project_root,
        "run_name": run_name,
        "runs_dir": runs_dir,
        "run_dir": run_dir,
        "tfrecord": resolve_from_project_root(project_root, paths_cfg["tfrecord"]),
        "manifest": resolve_from_project_root(project_root, paths_cfg["manifest"]),
        "split_source_dir": split_source_dir,
        "source_train_indices": split_source_dir / "train_meta_indices.txt",
        "source_val_indices": split_source_dir / "val_meta_indices.txt",
        "source_test_indices": split_source_dir / "test_meta_indices.txt",
        "split_dir": split_dir,
        "train_indices": split_dir / "train_meta_indices.txt",
        "val_indices": split_dir / "val_meta_indices.txt",
        "test_indices": split_dir / "test_meta_indices.txt",
        "artifacts_dir": artifacts_dir,
        "results_dir": results_dir
        }
    return resolved

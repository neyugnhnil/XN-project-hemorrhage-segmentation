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

# bundle resolved paths into dictionary for the whole pipeline
def get_resolved_paths(cfg: dict[str, Any], config_path: str | Path) -> dict[str, Path]:
    project_root = resolve_project_root(cfg, config_path)
    paths_cfg = cfg["paths"]
    resolved = {
        "project_root": project_root,
        "tfrecord": resolve_from_project_root(project_root, paths_cfg["tfrecord"]),
        "manifest": resolve_from_project_root(project_root, paths_cfg["manifest"]),
        "train_indices": resolve_from_project_root(project_root, paths_cfg["train_indices"]),
        "val_indices": resolve_from_project_root(project_root, paths_cfg["val_indices"]),
        "test_indices": resolve_from_project_root(project_root, paths_cfg["test_indices"]),
        "artifacts_dir": resolve_from_project_root(project_root, paths_cfg["artifacts_dir"]),
        "results_dir": resolve_from_project_root(project_root, paths_cfg["results_dir"])
        }
    return resolved
#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Any
import pandas as pd

# io glue

def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_json(obj: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def write_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_parent_dir(path)
    df.to_csv(path, index=False)

def read_index_file(path: str | Path) -> list[int]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Index file not found: {path}")
    indices: list[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                indices.append(int(line))
            except ValueError as exc:
                raise ValueError(f"Invalid integer in index file {path} on line {line_number}: {line}") from exc
    return indices

def write_index_file(indices: list[int], path: str | Path) -> None:
    path = Path(path)
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        for idx in indices:
            f.write(f"{int(idx)}\n")
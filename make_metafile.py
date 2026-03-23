#!/usr/bin/env python3
"""
This script builds a metafile of usable CT scan IDs, classification labels, and segmentation 
labels from the contents of /data, based on inclusion rules.

outputs:
- /data/subset_metafile.csv
- /data/subset_report.json

rules:
- exclude IDs listed in flagged.txt
- within each render_directory, require the ID to be present across all 4 windows
- require the ID to exist in hemorrhage-labels.csv
- require the ID to exist in at least one segmentation results CSV
- require a usable segmentation polygon from human labeling platform:
- prefer Correct Label, otherwise use Majority Label
- if multiple segmentation rows exist for the same ID:
   - prefer row with Correct label over Majority label
   - prefer higher Agreement
   - prefer higher Total Qualified Reads
   - prefer higher Total Reads
   - prefer earlier source CSV name
   - prefer earlier Case ID
"""

import json
from pathlib import Path
import pandas as pd

## CONSTANTS ##
DATA_ROOT = Path("data")
RENDERS_DIR = DATA_ROOT / "renders"
SEG_DIR = DATA_ROOT / "segmentation"
LABELS_CSV = DATA_ROOT / "hemorrhage-labels.csv"
FLAGGED_TXT = DATA_ROOT / "flagged.txt"

OUT_METAFILE = DATA_ROOT / "subset_metafile.csv"
OUT_REPORT = DATA_ROOT / "subset_report.json"

WINDOWS = ["brain_bone_window","brain_window","max_contrast_window","subdural_window"]
CLASS_DIRS = ["epidural","intraparenchymal","intraventricular","multi","normal","subarachnoid","subdural"]
CLASS_LABEL_COLUMNS = ["any","epidural","intraparenchymal","intraventricular","subarachnoid","subdural"]

## SMALL HELPERS ##

def normalize_id(value) -> str:
    # `ID____.jpg` -> `ID____`
    s = str(value).strip()
    if s.endswith(".jpg"):
        s = s[:-4]
    return s

def normalize_id_series(series: pd.Series) -> pd.Series:
    # vectorized version of normalize_id for pandas columns
    return (series.astype("string").str.strip().str.replace(r"\.jpg$", "", regex=True))

def is_empty_label(series: pd.Series) -> pd.Series:
    # is this pandas field empty? (including NaN, None, null etc)
    s = series.astype("string").str.strip()
    return s.isna() | s.isin(["", "[]", "nan", "None", "null"])

def safe_numeric(series: pd.Series) -> pd.Series:
    # converts pandas column into numerics, bad values become NaN
    return pd.to_numeric(series, errors="coerce")

## FILE READERS ##

def read_flagged_ids() -> set[str]:
    # retrieves flagged ids set. it will later be excluded globally.
    path = FLAGGED_TXT
    if not path.exists():
        return set()
    
    flagged_ids = set()
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            flagged_ids.add(normalize_id(line))
    
    return flagged_ids

def read_classification_labels() -> pd.DataFrame:

    usecols = ["Image"] + CLASS_LABEL_COLUMNS # columns to read in the class csv

    dtype = { # explicit dtypes dict for pandas
        "Image": "string", 
        **{col: "Int64" for col in CLASS_LABEL_COLUMNS}
        }

    df = pd.read_csv(LABELS_CSV, usecols=usecols, dtype=dtype) 
    
    df["id"] = normalize_id_series(df["Image"])

    return (
        df[["id"] + CLASS_LABEL_COLUMNS]
        .drop_duplicates(subset=["id"], keep="first") # keep the first one if id appears twice
    )

def read_best_segmentation_rows():
    # 1. determine which IDs have usable segmentation labels
    # 2. if an ID has multiple usable rows, choose the best one

    csv_paths = sorted(SEG_DIR.glob("results_*.csv")) # sorted for deterministic tie breaking

    candidate_frames = [] # to contain all usable rows
    all_seg_ids = set() # to contain all IDs encountered (for report)

    report_counts = { # for json report
        "segmentation_rows_total": 0,
        "segmentation_rows_with_correct": 0,
        "segmentation_rows_with_majority_only": 0,
        "segmentation_rows_without_usable_polygon": 0
        }

    needed_cols = [ # list of columns to read from csvs
        "Origin",
        "Correct Label",
        "Majority Label",
        "Agreement",
        "Total Qualified Reads",
        "Total Reads",
        "Case ID"
        ]

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, usecols=lambda c: c in needed_cols).copy()

        # make sure ranking columns exist even if missing from some CSVs
        for col in ["Agreement", "Total Qualified Reads", "Total Reads", "Case ID"]:
            if col not in df.columns:
                df[col] = pd.NA

        # normalized id column based on Origin
        df["id"] = normalize_id_series(df["Origin"])

        # add all ids from this csv to all_seg_ids
        all_seg_ids.update(df["id"].dropna().astype(str).tolist())

        # boolean masks for correct/majority label presence
        correct_present = ~is_empty_label(df["Correct Label"])
        majority_present = ~is_empty_label(df["Majority Label"])
        usable_mask = correct_present | majority_present

        # counts for the report
        report_counts["segmentation_rows_total"] += int(len(df))
        report_counts["segmentation_rows_with_correct"] += int(correct_present.sum())
        report_counts["segmentation_rows_with_majority_only"] += int((~correct_present & majority_present).sum())
        report_counts["segmentation_rows_without_usable_polygon"] += int((~usable_mask).sum())

        # filters df to only usable rows
        usable = df.loc[usable_mask].copy()
        if usable.empty:
            continue

        # default to majority label for all usable rows
        usable["seg_label_source"] = "majority"
        usable["seg_polygon"] = usable["Majority Label"]

        # override with correct label where present
        usable_has_correct = correct_present.loc[usable_mask]
        usable.loc[usable_has_correct, "seg_label_source"] = "correct"
        usable.loc[usable_has_correct, "seg_polygon"] = usable.loc[usable_has_correct, "Correct Label"]

        # convert all ranking fields to sortable series
        usable["source_rank"] = usable["seg_label_source"].map({"correct": 0, "majority": 1}).fillna(9)
        usable["Agreement"] = safe_numeric(usable["Agreement"]).fillna(-1)
        usable["Total Qualified Reads"] = safe_numeric(usable["Total Qualified Reads"]).fillna(-1)
        usable["Total Reads"] = safe_numeric(usable["Total Reads"]).fillna(-1)
        usable["seg_source_csv"] = csv_path.name
        usable["Case ID Sort"] = usable["Case ID"].astype("string").fillna("")
        
        # append usable df to candidate_frames (only columns needed downstream)
        candidate_frames.append(
            usable[["id","seg_label_source","seg_polygon",
            "Agreement","Total Qualified Reads","Total Reads",
            "Case ID Sort","seg_source_csv","source_rank"]]
            )

    # stack all the usable dfs into one
    candidates = pd.concat(candidate_frames, ignore_index=True)

    # sort based on ranking defined in per-csv for loop above
    candidates = candidates.sort_values(
        by=["id","source_rank","Agreement","Total Qualified Reads",
        "Total Reads","seg_source_csv","Case ID Sort"],
        ascending=[True, True, False, False, False, True, True],
    )

    best = (
        candidates.drop_duplicates(subset=["id"], keep="first")[
            ["id", "seg_label_source", "seg_polygon"]
            ]
            .reset_index(drop=True)
    )

    return best, all_seg_ids, report_counts

## RENDER COMPLETENESS CHECK ##

def build_render_inventory() -> pd.DataFrame:
    # returns a pandas df of every ct scan image 

    rows = []
    for render_directory in CLASS_DIRS:
        # all render_directory directories under /render
        class_root = RENDERS_DIR / render_directory

        for window in WINDOWS:
            # all windows under render_directory/render/
            window_root = class_root / window

            for img_path in window_root.glob("*.jpg"):
                # all files ending in jpg
                rows.append((img_path.stem, render_directory, window, str(img_path)))

    return pd.DataFrame(rows, columns=["id", "render_directory", "window", "path"])

def build_render_completeness(render_df: pd.DataFrame) -> pd.DataFrame:
    # check for each (render_directory, id) pair whether all windows are present

    expected_number_of_windows = len(WINDOWS)

    grouped = (
        render_df.groupby(["render_directory", "id"])["window"]   # group by hemorragetype, id
        .agg(num_windows_present="nunique")                     # aggregate by unique window values
        .reset_index()                                          # convert GroupBy index into normal columns
    )

    # create has_all_windows column based on num_windows_present
    grouped["has_all_windows"] = grouped["num_windows_present"] == expected_number_of_windows
    return grouped

## RENDER PATH COLUMNS ##

def pivot_render_paths(render_df: pd.DataFrame) -> pd.DataFrame:
    # makes pd df that groups render paths by [hemmorrhagetype, id] index

    pivot = (
        render_df.pivot_table(
            index=["render_directory", "id"],  # index by render_directory and id
            columns="window", # unique values in windows become column names
            values="path", # values come from path column
            aggfunc="first", # if there are instances of the same [index] and window, take first one
        )
        .reset_index()
    )

    return pivot.rename(
        columns=
        {
            "brain_bone_window": "brain_bone_window_path",
            "brain_window": "brain_window_path",
            "max_contrast_window": "max_contrast_window_path",
            "subdural_window": "subdural_window_path"
            }
        )

## MAIN() ##

def main() -> None:
    # read all source data
    flagged_ids = read_flagged_ids()
    render_df = build_render_inventory()
    render_completeness_df = build_render_completeness(render_df)

    cls_df = read_classification_labels()
    best_seg_df, all_seg_ids, seg_report_counts = read_best_segmentation_rows()

    # filter for valid (render_directory, id) 
    has_all_render_windows_df = render_completeness_df.loc[
        render_completeness_df["has_all_windows"],
        ["render_directory", "id"]
        ].copy()

    has_cls_and_seg_df = (
        cls_df[["id"]]
        .merge(best_seg_df[["id"]], on="id", how="inner")
        .drop_duplicates()
        )
    
    filtered_ids_df = has_all_render_windows_df.merge(has_cls_and_seg_df, on="id", how="inner")
    
    # drop flagged
    filtered_ids_df = filtered_ids_df.loc[~filtered_ids_df["id"].isin(flagged_ids)].copy()

    if filtered_ids_df.empty:
        raise RuntimeError("No rows remain after filtering.")

    # build metafile
    filtered_render_df = render_df.merge(filtered_ids_df,on=["render_directory", "id"],how="inner")
    render_paths_df = pivot_render_paths(filtered_render_df)
    meta_df = (render_paths_df\
        .merge(cls_df, on="id", how="inner")\
            .merge(best_seg_df, on="id", how="inner"))

    out_cols = [
        "id",
        "render_directory",
        "any",
        "epidural",
        "intraparenchymal",
        "intraventricular",
        "subarachnoid",
        "subdural",
        "seg_label_source",
        "seg_polygon",
        "brain_bone_window_path",
        "brain_window_path",
        "max_contrast_window_path",
        "subdural_window_path",
    ]

    meta_df = meta_df[out_cols].sort_values(["render_directory", "id"]).reset_index(drop=True)
    meta_df.to_csv(OUT_METAFILE, index=False)

    # success report
    cls_ids = set(cls_df["id"].astype(str))
    usable_seg_ids = set(best_seg_df["id"].astype(str))

    report = {
        "paths": {
            "metafile_csv": str(OUT_METAFILE),
        },
        "counts": {
            "render_images_total": int(len(render_df)),
            "classification_ids_total": int(len(cls_ids)),
            "segmentation_ids_total_any_row": int(len(all_seg_ids)),
            "segmentation_ids_total_usable": int(len(usable_seg_ids)),
            "final_rows": int(len(meta_df)),
            "final_unique_ids": int(meta_df["id"].nunique()),
        },
        "exclusions": {
            "id_in_flagged": int(render_completeness_df["id"].isin(flagged_ids).sum()),
            "(id,render_directory)_with_missing_windows": int((~render_completeness_df["has_all_windows"]).sum()),
            "id_with_missing_class_label": int((~render_completeness_df["id"].isin(cls_ids)).sum()),
            "id_without_any_seg_labels": int((~render_completeness_df["id"].isin(all_seg_ids)).sum()),
            "id_without_usable_seg_labels": int(
                render_completeness_df["id"].isin(all_seg_ids).sum()
                - render_completeness_df["id"].isin(usable_seg_ids).sum()
            ),
        },
        "final_metafile_summary": seg_report_counts,
        "final_rows_by_render_directory": {
            str(k): int(v)
            for k, v in meta_df["render_directory"].value_counts().sort_index().to_dict().items()
        },
        "seg_label_source_counts": {
            str(k): int(v)
            for k, v in meta_df["seg_label_source"].value_counts().sort_index().to_dict().items()
        },
    }

    OUT_REPORT.write_text(json.dumps(report, indent=2))
    print(f"Wrote metafile: {OUT_METAFILE}")
    print(f"Wrote report: {OUT_REPORT}")


if __name__ == "__main__":
    main()
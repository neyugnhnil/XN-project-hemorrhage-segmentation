import sys

BASE_MODEL_FAMILIES = ["mlp", "cnn", "multiunet"]
OPTIONAL_MODEL_FAMILIES = ["attentioncnn"]
SUPPORTED_MODEL_FAMILIES = BASE_MODEL_FAMILIES + OPTIONAL_MODEL_FAMILIES

MODEL_FAMILIES = list(
    config.get("workflow", {}).get("model_families", SUPPORTED_MODEL_FAMILIES)
)

UNKNOWN_MODEL_FAMILIES = sorted(set(MODEL_FAMILIES) - set(SUPPORTED_MODEL_FAMILIES))
if UNKNOWN_MODEL_FAMILIES:
    raise ValueError(
        "workflow.model_families contains unsupported models: "
        + ", ".join(UNKNOWN_MODEL_FAMILIES)
    )

MISSING_BASE_MODEL_FAMILIES = sorted(set(BASE_MODEL_FAMILIES) - set(MODEL_FAMILIES))
if MISSING_BASE_MODEL_FAMILIES:
    raise ValueError(
        "workflow.model_families must include the baseline models: "
        + ", ".join(MISSING_BASE_MODEL_FAMILIES)
    )

ATTENTIONCNN_ENABLED = "attentioncnn" in MODEL_FAMILIES

RUN_NAME = str(config.get("run_name", "default")).strip()
if not RUN_NAME:
    raise ValueError("run_name must not be empty")
if "/" in RUN_NAME or "\\" in RUN_NAME:
    raise ValueError("run_name must be a plain folder name, not a path")

RUNS_DIR = config["paths"].get("runs_dir", "runs")
RUN_DIR = f"{RUNS_DIR}/{RUN_NAME}"
SPLIT_DIR = f"{RUN_DIR}/splits"
ARTIFACT_DIR = f"{RUN_DIR}/artifacts"
RESULTS_DIR = f"{RUN_DIR}/results"

LOG_DIR = f"{ARTIFACT_DIR}/logs"
MODEL_DIR = f"{ARTIFACT_DIR}/models"
CALIB_DIR = f"{ARTIFACT_DIR}/calibration"
SPLIT_SUMMARY = f"{ARTIFACT_DIR}/splits/split_summary.json"
ATTENTIONCNN_SEGMENTER_WEIGHTS = f"{MODEL_DIR}/attentioncnn/attentioncnn_segmenter_weights.keras"
ATTENTIONCNN_ALL_TARGETS = [ATTENTIONCNN_SEGMENTER_WEIGHTS] if ATTENTIONCNN_ENABLED else []
PYTHON = sys.executable

TRAIN_INDICES = f"{SPLIT_DIR}/train_meta_indices.txt"
VAL_INDICES = f"{SPLIT_DIR}/val_meta_indices.txt"
TEST_INDICES = f"{SPLIT_DIR}/test_meta_indices.txt"


def train_script_path(model_family):
    return f"scripts/train_{model_family}.py"

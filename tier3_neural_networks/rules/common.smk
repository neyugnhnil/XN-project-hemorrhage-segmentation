MODEL_FAMILIES = ["mlp", "cnn", "multiunet"]

ARTIFACT_DIR = config["paths"]["artifacts_dir"]
RESULTS_DIR = config["paths"]["results_dir"]

LOG_DIR = f"{ARTIFACT_DIR}/logs"
MODEL_DIR = f"{ARTIFACT_DIR}/models"
CALIB_DIR = f"{ARTIFACT_DIR}/calibration"
SPLIT_SUMMARY = f"{ARTIFACT_DIR}/splits/split_summary.json"


def train_script_path(model_family):
    return f"scripts/train_{model_family}.py"
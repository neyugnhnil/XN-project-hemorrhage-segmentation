ATTENTIONCNN_EVAL_INPUTS = (
    [
        f"{MODEL_DIR}/attentioncnn/attentioncnn_weights.keras",
        ATTENTIONCNN_SEGMENTER_WEIGHTS,
        f"{CALIB_DIR}/attentioncnn/attentioncnn_thresholds.json",
    ]
    if ATTENTIONCNN_ENABLED
    else []
)

ATTENTIONCNN_EVAL_ARGS = (
    f"--attentioncnn-weights {MODEL_DIR}/attentioncnn/attentioncnn_weights.keras "
    f"--attentioncnn-segmenter-weights {ATTENTIONCNN_SEGMENTER_WEIGHTS} "
    f"--attentioncnn-thresholds {CALIB_DIR}/attentioncnn/attentioncnn_thresholds.json"
    if ATTENTIONCNN_ENABLED
    else ""
)


rule evaluate_all:
    input:
        config="config.yaml",
        test_indices=TEST_INDICES,
        mlp_weights=f"{MODEL_DIR}/mlp/mlp_weights.keras",
        mlp_thresholds=f"{CALIB_DIR}/mlp/mlp_thresholds.json",
        cnn_weights=f"{MODEL_DIR}/cnn/cnn_weights.keras",
        cnn_thresholds=f"{CALIB_DIR}/cnn/cnn_thresholds.json",
        multiunet_weights=f"{MODEL_DIR}/multiunet/multiunet_weights.keras",
        multiunet_thresholds=f"{CALIB_DIR}/multiunet/multiunet_thresholds.json",
        attentioncnn=ATTENTIONCNN_EVAL_INPUTS
    output:
        metrics_csv=f"{RESULTS_DIR}/test_metrics_all_models.csv",
        predictions_csv=f"{RESULTS_DIR}/test_predictions_all_models.csv",
        segmentation_csv=f"{RESULTS_DIR}/segmentation_metrics_all_models.csv",
        summary_json=f"{RESULTS_DIR}/evaluation_summary.json"
    log:
        f"{LOG_DIR}/evaluate_all.log"
    threads: 1
    resources:
        gpu=1
    params:
        attentioncnn_args=ATTENTIONCNN_EVAL_ARGS
    shell:
        r"""
        mkdir -p {RESULTS_DIR} {LOG_DIR}
        {PYTHON} scripts/evaluate_all.py \
            --config {input.config} \
            --test-indices {input.test_indices} \
            --mlp-weights {input.mlp_weights} \
            --mlp-thresholds {input.mlp_thresholds} \
            --cnn-weights {input.cnn_weights} \
            --cnn-thresholds {input.cnn_thresholds} \
            --multiunet-weights {input.multiunet_weights} \
            --multiunet-thresholds {input.multiunet_thresholds} \
            {params.attentioncnn_args} \
            --metrics-out {output.metrics_csv} \
            --predictions-out {output.predictions_csv} \
            --segmentation-out {output.segmentation_csv} \
            --summary-out {output.summary_json} \
            > {log} 2>&1
        """


rule all:
    default_target: True
    input:
        TRAIN_INDICES,
        VAL_INDICES,
        TEST_INDICES,
        SPLIT_SUMMARY,

        ATTENTIONCNN_ALL_TARGETS,
        expand(f"{MODEL_DIR}/{{model_family}}/{{model_family}}_weights.keras", model_family=MODEL_FAMILIES),
        expand(f"{MODEL_DIR}/{{model_family}}/{{model_family}}_train_summary.json", model_family=MODEL_FAMILIES),

        expand(f"{CALIB_DIR}/{{model_family}}/{{model_family}}_thresholds.json", model_family=MODEL_FAMILIES),
        expand(f"{CALIB_DIR}/{{model_family}}/{{model_family}}_calibration_summary.json", model_family=MODEL_FAMILIES),

        f"{RESULTS_DIR}/test_metrics_all_models.csv",
        f"{RESULTS_DIR}/test_predictions_all_models.csv",
        f"{RESULTS_DIR}/segmentation_metrics_all_models.csv",
        f"{RESULTS_DIR}/evaluation_summary.json"

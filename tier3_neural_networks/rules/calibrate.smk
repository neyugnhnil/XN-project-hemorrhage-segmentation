rule calibrate_model:
    input:
        config="config.yaml",
        weights=f"{MODEL_DIR}/{{model_family}}/{{model_family}}_weights.keras",
        train_summary=f"{MODEL_DIR}/{{model_family}}/{{model_family}}_train_summary.json",
        val_indices=VAL_INDICES
    output:
        thresholds=f"{CALIB_DIR}/{{model_family}}/{{model_family}}_thresholds.json",
        summary=f"{CALIB_DIR}/{{model_family}}/{{model_family}}_calibration_summary.json"
    log:
        f"{LOG_DIR}/calibrate_{{model_family}}.log"
    threads: 1
    resources:
        gpu=1
    wildcard_constraints:
        model_family="mlp|cnn|multiunet"
    shell:
        r"""
        mkdir -p {CALIB_DIR}/{wildcards.model_family} {LOG_DIR}
        {PYTHON} scripts/calibrate_thresholds.py \
            --config {input.config} \
            --model-family {wildcards.model_family} \
            --weights {input.weights} \
            --val-indices {input.val_indices} \
            --thresholds-out {output.thresholds} \
            --summary-out {output.summary} \
            > {log} 2>&1
        """


rule calibrate_attentioncnn:
    input:
        config="config.yaml",
        weights=f"{MODEL_DIR}/attentioncnn/attentioncnn_weights.keras",
        segmenter_weights=ATTENTIONCNN_SEGMENTER_WEIGHTS,
        train_summary=f"{MODEL_DIR}/attentioncnn/attentioncnn_train_summary.json",
        val_indices=VAL_INDICES
    output:
        thresholds=f"{CALIB_DIR}/attentioncnn/attentioncnn_thresholds.json",
        summary=f"{CALIB_DIR}/attentioncnn/attentioncnn_calibration_summary.json"
    log:
        f"{LOG_DIR}/calibrate_attentioncnn.log"
    threads: 1
    resources:
        gpu=1
    shell:
        r"""
        mkdir -p {CALIB_DIR}/attentioncnn {LOG_DIR}
        {PYTHON} scripts/calibrate_thresholds.py \
            --config {input.config} \
            --model-family attentioncnn \
            --weights {input.weights} \
            --segmenter-weights {input.segmenter_weights} \
            --val-indices {input.val_indices} \
            --thresholds-out {output.thresholds} \
            --summary-out {output.summary} \
            > {log} 2>&1
        """

rule calibrate_model:
    input:
        config="config.yaml",
        weights=f"{MODEL_DIR}/{{model_family}}/{{model_family}}_weights.keras",
        train_summary=f"{MODEL_DIR}/{{model_family}}/{{model_family}}_train_summary.json",
        val_indices=config["paths"]["val_indices"]
    output:
        thresholds=f"{CALIB_DIR}/{{model_family}}/{{model_family}}_thresholds.json",
        summary=f"{CALIB_DIR}/{{model_family}}/{{model_family}}_calibration_summary.json"
    log:
        f"{LOG_DIR}/calibrate_{{model_family}}.log"
    wildcard_constraints:
        model_family="mlp|cnn|multiunet"
    shell:
        r"""
        mkdir -p {CALIB_DIR}/{wildcards.model_family} {LOG_DIR}
        python scripts/calibrate_thresholds.py \
            --config {input.config} \
            --model-family {wildcards.model_family} \
            --weights {input.weights} \
            --val-indices {input.val_indices} \
            --thresholds-out {output.thresholds} \
            --summary-out {output.summary} \
            > {log} 2>&1
        """
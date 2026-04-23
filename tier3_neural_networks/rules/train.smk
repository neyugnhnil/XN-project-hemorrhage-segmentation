rule train_model:
    input:
        config="config.yaml",
        train_indices=TRAIN_INDICES,
        val_indices=VAL_INDICES,
        script="scripts/train_{model_family}.py"
    output:
        weights=f"{MODEL_DIR}/{{model_family}}/{{model_family}}_weights.keras",
        summary=f"{MODEL_DIR}/{{model_family}}/{{model_family}}_train_summary.json"
    log:
        f"{LOG_DIR}/train_{{model_family}}.log"
    threads: 1
    resources:
        gpu=1
    wildcard_constraints:
        model_family="mlp|cnn|multiunet"
    shell:
        r"""
        set -o pipefail
        mkdir -p {MODEL_DIR}/{wildcards.model_family} {LOG_DIR}
        PYTHONUNBUFFERED=1 {PYTHON} -u {input.script} \
            --config {input.config} \
            --train-indices {input.train_indices} \
            --val-indices {input.val_indices} \
            --weights-out {output.weights} \
            --summary-out {output.summary} \
            2>&1 | tee {log}
        """


rule train_attentioncnn:
    input:
        config="config.yaml",
        train_indices=TRAIN_INDICES,
        val_indices=VAL_INDICES,
        script="scripts/train_attentioncnn.py"
    output:
        weights=f"{MODEL_DIR}/attentioncnn/attentioncnn_weights.keras",
        segmenter_weights=ATTENTIONCNN_SEGMENTER_WEIGHTS,
        summary=f"{MODEL_DIR}/attentioncnn/attentioncnn_train_summary.json"
    log:
        f"{LOG_DIR}/train_attentioncnn.log"
    threads: 1
    resources:
        gpu=1
    shell:
        r"""
        set -o pipefail
        mkdir -p {MODEL_DIR}/attentioncnn {LOG_DIR}
        PYTHONUNBUFFERED=1 {PYTHON} -u {input.script} \
            --config {input.config} \
            --train-indices {input.train_indices} \
            --val-indices {input.val_indices} \
            --weights-out {output.weights} \
            --segmenter-weights-out {output.segmenter_weights} \
            --summary-out {output.summary} \
            2>&1 | tee {log}
        """

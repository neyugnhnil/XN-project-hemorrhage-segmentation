rule train_model:
    input:
        config="config.yaml",
        train_indices=config["paths"]["train_indices"],
        val_indices=config["paths"]["val_indices"],
        script="scripts/train_{model_family}.py"
    output:
        weights=f"{MODEL_DIR}/{{model_family}}/{{model_family}}_weights.keras",
        summary=f"{MODEL_DIR}/{{model_family}}/{{model_family}}_train_summary.json"
    log:
        f"{LOG_DIR}/train_{{model_family}}.log"
    wildcard_constraints:
        model_family="mlp|cnn|multiunet"
    shell:
        r"""
        mkdir -p {MODEL_DIR}/{wildcards.model_family} {LOG_DIR}
        python {input.script} \
            --config {input.config} \
            --train-indices {input.train_indices} \
            --val-indices {input.val_indices} \
            --weights-out {output.weights} \
            --summary-out {output.summary} \
            > {log} 2>&1
        """
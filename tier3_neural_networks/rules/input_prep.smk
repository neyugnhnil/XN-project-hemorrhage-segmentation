rule input_prep:
    input:
        config="config.yaml"
    output:
        train_indices=config["paths"]["train_indices"],
        val_indices=config["paths"]["val_indices"],
        test_indices=config["paths"]["test_indices"],
        split_summary=SPLIT_SUMMARY
    log:
        f"{LOG_DIR}/input_prep.log"
    shell:
        r"""
        mkdir -p splits {ARTIFACT_DIR}/splits {LOG_DIR}
        python scripts/input_prep.py \
            --config {input.config} \
            > {log} 2>&1
        """
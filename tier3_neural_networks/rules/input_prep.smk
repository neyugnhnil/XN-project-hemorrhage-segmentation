rule input_prep:
    input:
        config="config.yaml"
    output:
        train_indices=TRAIN_INDICES,
        val_indices=VAL_INDICES,
        test_indices=TEST_INDICES,
        split_summary=SPLIT_SUMMARY
    log:
        f"{LOG_DIR}/input_prep.log"
    shell:
        r"""
        mkdir -p {SPLIT_DIR} {ARTIFACT_DIR}/splits {LOG_DIR}
        {PYTHON} scripts/input_prep.py \
            --config {input.config} \
            > {log} 2>&1
        """

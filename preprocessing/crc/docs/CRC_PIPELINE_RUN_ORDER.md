# CRC Pipeline Run Order

Recommended Terminal sequence should be followed one checkpoint at a time:

1. `stage2-validate`
2. unit tests
3. `--fast-test --validate-only`
4. `--fast-test --run-primary`
5. review fast-test outputs
6. full primary modelling
7. post-run diagnostics
8. review diagnostics
9. downstream stages

The primary modelling run should not be followed by biological interpretation until post-run diagnostics confirm that selected models, frozen thresholds, probability columns, feature order and metric recomputation are correct.

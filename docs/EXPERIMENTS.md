# Experiment protocol

## Reproduce

1. Install the project in a fresh Python 3.11 environment with `python -m pip install -e ".[dev]"`.
2. Run `python -m movie_recs train` from the repository root.
3. Inspect `reports/metrics.json` and the generated figures. The report records seed, Python version, installed library versions and dataset source information.
4. Run `python -m movie_recs predict --input examples/sample.json` to exercise the trained model.
5. Run `make check test` to verify the contracts.

## Selection rule

A global timestamp split creates approximately 80%/10%/10% train/validation/test partitions; equal timestamps remain together. The training catalog contains only items with training interactions. A 16-dimensional factor model minimizes observed squared rating error plus L2 regularization with analytical gradients and a fixed 100-iteration L-BFGS-B budget. Validation NDCG@10 chooses factorization or popularity for inference. No retraining on validation occurs. Rating RMSE uses warm user-item rows; ranking treats ratings ≥ 4 as relevant, excludes already consumed titles, and ranks every eligible training-catalog title without sampled negatives.

## Reading the result

Popularity wins validation NDCG and remains the selected inference strategy, even though factorization has higher test NDCG on warm users. The global split produces substantial cold start: only 1,344 of 10,000 test ratings involve warm users and known items. Warm ranking covers 52 users; the additional all-user report covers 147 eligible users and uses popularity for unseen users. The optimizer hits its 100-iteration cap; the report explicitly records that it did not declare convergence.

The committed results are one seeded benchmark run. They do not establish confidence intervals, statistical significance, robustness across random seeds or production behavior. CV standard deviations summarize folds, not uncertainty on future production metrics. Do not tune settings against the final test set; use new validation experiments and reserve a fresh holdout if making repeated test-driven changes.

## Notebook

`notebooks/01_review_results.ipynb` reads committed reports and displays the evaluation figure. It is executed and can be viewed directly on GitHub without training. To rerun it locally, install `python -m pip install -e ".[notebook]"` and select that environment in your notebook editor. Training logic remains in the package so notebook execution cannot silently diverge from the command-line experiment.

## Resource boundaries

All examples run on CPU without cloud accounts or paid APIs. The default dataset archives are small, and raw data is cached locally. The CNN uses two CPU threads; the wine forest uses two workers. Reports and small images are committed, while model binaries, raw archives and environment folders are ignored. CI has a 15-minute execution limit.

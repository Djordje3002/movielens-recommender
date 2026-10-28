# Model card — MovieLens Recommender

Version: 1.0.0 · Experiment seed: 42 · Selected model: `popularity`

## Intended use

Rank unseen movies from historical user ratings and measure whether personalized collaborative filtering adds value over popular titles. The project implements regularized biased matrix factorization directly, then evaluates both rating error and full-catalog recommendation quality.

The intended audience is engineers reviewing a reproducible ML portfolio and learners inspecting model selection, evaluation and inference. The project has no claimed deployment history or customer results.

## Training and evaluation

A global timestamp split creates approximately 80%/10%/10% train/validation/test partitions; equal timestamps remain together. The training catalog contains only items with training interactions. A 16-dimensional factor model minimizes observed squared rating error plus L2 regularization with analytical gradients and a fixed 100-iteration L-BFGS-B budget. Validation NDCG@10 chooses factorization or popularity for inference. No retraining on validation occurs. Rating RMSE uses warm user-item rows; ranking treats ratings ≥ 4 as relevant, excludes already consumed titles, and ranks every eligible training-catalog title without sampled negatives.

| Model | recall at 10 | ndcg at 10 | catalog coverage at 10 |
| --- | ---: | ---: | ---: |
| `popularity` | 0.0550 | 0.1160 | 0.0470 |
| `factorization` | 0.0872 | 0.1374 | 0.1027 |

**Warm user-item rating prediction:**

| Model | mae | rmse | r2 |
| --- | ---: | ---: | ---: |
| `global_mean_baseline` | 1.0414 | 1.1951 | -0.0321 |
| `factorization` | 0.8070 | 0.9997 | 0.2779 |

**All eligible users, including cold-start popularity fallback:**

| Model | recall at 10 | ndcg at 10 | catalog coverage at 10 |
| --- | ---: | ---: | ---: |
| `popularity` | 0.0790 | 0.2268 | 0.0470 |
| `factorization` | 0.0904 | 0.2344 | 0.1095 |


## Interpretation

Popularity wins validation NDCG and remains the selected inference strategy, even though factorization has higher test NDCG on warm users. The global split produces substantial cold start: only 1,344 of 10,000 test ratings involve warm users and known items. Warm ranking covers 52 users; the additional all-user report covers 147 eligible users and uses popularity for unseen users. The optimizer hits its 100-iteration cap; the report explicitly records that it did not declare convergence.

## Limitations

Historical explicit ratings have selection bias and do not represent randomized exposure. A missing rating is not a known dislike, and offline relevance does not establish user satisfaction. The small warm-user cohort makes ranking metrics noisy. New items are excluded because collaborative factors cannot score them reliably. There is no content model, online A/B test, or causal claim. Complexity is roughly O(iterations × ratings × factors) for training and O(catalog × factors) per warm-user ranking.

## Artifacts and maintenance

Run the training command to rebuild local artifacts and reports. Load only your own trusted local artifacts. For any data or model change, rerun training, compare with the baseline under the same split protocol, and inspect failure cases. A changed dataset hash or split constitutes a different experiment and must be documented as such. Fresh labeled data is required to estimate real deployment performance.

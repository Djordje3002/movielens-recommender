# Data card

MovieLens 100K from GroupLens: 100,000 ratings, 943 users, 1,682 movies. Only user ID, movie ID, rating and timestamp are loaded. Demographic files are not used. The training catalog in this split contains 1,616 items.

## Source and citation

[Original dataset](https://grouplens.org/datasets/movielens/100k/)

Harper, F. Maxwell & Konstan, Joseph A. (2015). The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems 5(4), Article 19. https://doi.org/10.1145/2827872

## License

MovieLens 100K has separate research-use terms. Do not redistribute its data without permission, imply GroupLens endorsement, or use it commercially without the required permission. Review the [original usage license](https://files.grouplens.org/datasets/movielens/ml-100k-README.txt) before use. No raw data or learned user factors are included in this repository.

## Integrity and storage

```text
{
  "url": "https://files.grouplens.org/datasets/movielens/ml-100k.zip",
  "sha256": "50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229"
}
```

For downloaded datasets, every run verifies archive bytes against this SHA-256 before parsing. The downloader has a 60-second timeout and 20 MB response cap. It reads named members without extracting filesystem paths. Raw files stay in the ignored `data/raw/` cache. To recover from a corrupted cache, remove `data/raw/source.zip`, verify the upstream source, then rerun training. Do not silently replace the expected hash after a source change.

## Processing and sampling

A global timestamp split creates approximately 80%/10%/10% train/validation/test partitions; equal timestamps remain together. The training catalog contains only items with training interactions. A 16-dimensional factor model minimizes observed squared rating error plus L2 regularization with analytical gradients and a fixed 100-iteration L-BFGS-B budget. Validation NDCG@10 chooses factorization or popularity for inference. No retraining on validation occurs. Rating RMSE uses warm user-item rows; ranking treats ratings ≥ 4 as relevant, excludes already consumed titles, and ranks every eligible training-catalog title without sampled negatives.

## Representation and privacy

Historical explicit ratings have selection bias and do not represent randomized exposure. A missing rating is not a known dislike, and offline relevance does not establish user satisfaction. The small warm-user cohort makes ranking metrics noisy. New items are excluded because collaborative factors cannot score them reliably. There is no content model, online A/B test, or causal claim. Complexity is roughly O(iterations × ratings × factors) for training and O(catalog × factors) per warm-user ranking.

No private account data, API credentials, or scraped personal profiles are used. Synthetic fixtures exist only in tests and manually constructed examples; published metrics come from the cited dataset.

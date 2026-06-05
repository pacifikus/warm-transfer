# warm-transfer

A **post-hoc, model-agnostic** library for **item cold-start** recommendation.

It wraps an *already-trained* recommender (BPR, etc.) as a black box and predicts
scores for **new (cold) items** from their content features — **without retraining
the base model**. The project is a library + honest benchmark across method
families, not a single champion method.

See [`RESEARCH.md`](RESEARCH.md) for the full framing, methodology, and findings
(including the §6.1 feature-group ablation).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on Unix)
pip install -r requirements.txt
```

## Data

Raw datasets are **not** committed. Prepare them into the standard processed-CSV
format with the per-dataset scripts:

```bash
# MovieLens-1M: put ratings.dat / movies.dat in data/ml-1m/
python datasets/prepare_ml1m.py            # -> data/processed/

# Amazon Toys & Games: put the 5-core review + meta JSON in data/raw/amazon_toys/
python datasets/prepare_amazon_toys.py     # -> data/processed/amazon_toys/
```

Each script emits the standard pair (schema enforced by `validate_schema`):
- `interactions.csv` — `user_id, item_id, engagement` (+ optional `timestamp`)
- `item_features.csv` — `item_id` + numeric content-feature columns

## Run

```bash
python benchmarks/run.py --dataset ml1m         # main benchmark (ml1m | amazon)
python benchmarks/ablation.py --dataset ml1m    # calibrator feature-group ablation
```

Results print as a table and save to `benchmarks/*.csv` (git-ignored — reproduce
by re-running).

## Layout

```
warmtransfer/        core library (feature-agnostic)
  core/              Dataset, Model, ColdStartMethod/Context, BenchmarkRunner
  methods/           baselines, score/embedding aggregation, stacking calibrator
  models/            BPR base model
  similarity/        content item-similarity index
  metrics/ splitters/
datasets/            dataset registration + per-dataset preprocessing scripts
benchmarks/          run.py (benchmark), ablation.py (feature-group ablation)
RESEARCH.md          framing, methodology, results, decisions log
```

## Adding a dataset

If it is already in the processed-CSV format, one line in
[`datasets/__init__.py`](datasets/__init__.py):

```python
register("mydata", ProcessedDataset("My Data", "data/processed/mydata"))
```

Genuinely different loading logic subclasses `Dataset` (see
[`warmtransfer/core/dataset.py`](warmtransfer/core/dataset.py)).

# Root-Join-Experiments

Python implementation of Root Join for approximate k-nearest
neighbor self-joins, using the SISAP 2025 Task 2 datasets and evaluation
framework.

The repository explores the relationship between neighbor recall,
execution time, and memory usage through dimensionality reduction,
parameter sweeps, and parallel group processing.

It builds on the
[sisap25-example-python repository](https://github.com/sisap-challenges/sisap25-example-python)
and the
[SISAP-Challenge-2025 repository](https://github.com/Fa11ingDeep/SISAP-Challenge-2025),
following the algorithm described in the
[Root Join paper](https://www.sciencedirect.com/science/article/pii/S0306437920300211).

## Experimental pipeline

For each dataset and parameter configuration, the implementation:

1. Downloads the dataset and its ground truth if they are not available locally.
2. Standardizes the input vectors.
3. Applies PCA, selecting enough components to retain at least 80% of the variance of the standardized data.
4. Randomly selects centers and builds groups.
5. Searches for neighbors within each point's group and neighboring groups.
6. Processes groups in parallel using 8 worker processes.
7. Writes intermediate results to Parquet and final results to HDF5.
8. Evaluates neighbor recall against the provided ground truth.

Each point retrieves neighbors from the same dataset, excluding itself.

Recall evaluation is performed separately using `eval.py`.

## Getting started

Clone the repository:

```bash
git clone https://github.com/Fa11ingDeep/Root-Join-Experiments.git
cd Root-Join-Experiments
```

Install the Requirements with:

```bash
pip install -r requirements.txt
```

Run the default experiment:

```bash
python root_join.py
```

The default dataset is `gooaq`, and the supported task is `task2`.
The command runs all parameter combinations configured for that dataset.

To select a dataset explicitly:

```bash
python root_join.py --task task2 --dataset pubmed23
```

Datasets and ground-truth files are downloaded automatically into `data/`.

## Datasets and parameter configurations

The following configurations are defined in `datasets.py`:

| Dataset | c | k | t |
| --- | --- | --- | --- |
| `gooaq` | 1, 2, 3, 4 | 1, 5, 10, 20, 30, 40, 50 | 1, 3, 5 |
| `ccnews` | 1, 2, 3, 4 | 1, 5, 10, 20, 30, 40, 50 | 1, 3, 5 |
| `yahooaq` | 1, 2, 3, 4 | 1, 5, 10, 20, 30, 40, 50 | 1, 3, 5 |
| `pubmed23` | 1, 2, 3, 4 | 1, 5, 10, 20, 30, 40, 50 | 1, 3, 5 |

Parameters control:

- **c:** center sampling and group sizing. The implementation uses
  `c1 = c`, `c2 = 1/c`, and selects `floor(c × sqrt(n))` centers.
- **k:** number of neighbors returned per point.
- **t:** number of neighboring groups considered when expanding
  the candidate set.

Parameter grids are edited in `datasets.py`; they are not exposed
as command-line arguments.

## Sequential execution

`orchestrator.py` runs a configured list of commands sequentially:

```bash
python orchestrator.py
```

## Output files

The pipeline uses three output formats:

| Format | Purpose |
| --- | --- |
| Parquet | Intermediate neighbor results under `temporary/`. |
| HDF5 | Final neighbor IDs, scores, and experiment metadata under `results/`. |
| CSV | Evaluation summary produced by `eval.py`. |

Final result files follow this pattern:

```text
results/<dataset>/task2/root_join_<dataset>_<c1>_<c2>_<k>_<t>.h5
```

Each HDF5 file contains:

- `knns`: neighbor IDs.
- `dists`: corresponding scores.
- Metadata identifying the dataset, task, algorithm, parameters,
  and recorded times.

Memory measurements for dimensionality reduction and self-join execution
are stored in the `params` metadata.

## Evaluation

Evaluate the saved results with:

```bash
python eval.py results.csv
```

The evaluator compares predicted neighbors with the provided ground truth
and produces a CSV containing:

```text
dataset, task, algo, buildtime, querytime, k, params, recall
```

Recorded timing fields have the following scope:

- `buildtime`: dimensionality reduction and group construction.
- `querytime`: parallel neighbor search and result consolidation,
  including temporary-file processing.

## Repository structure

```text
Root-Join-Experiments/
├── README.md
├── datasets.py        # Dataset URLs, loading, and parameter grids
├── root_join.py       # Preprocessing, grouping, search, and result storage
├── orchestrator.py    # Sequential experiment execution
├── eval.py            # Ground-truth recall evaluation and CSV export
└── requirements.txt   # Pinned dependencies
```

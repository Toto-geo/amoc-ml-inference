# Machine learning

This directory turns the prepared ocean data into trained models, held-out
predictions and reconstruction metrics.

<details>
<summary><strong>1. Overview</strong></summary>

The machine learning pipeline follows one fixed sequence.

1. **Load the prepared data.** Read the transport target, virtual Argo density
   profiles and four gridded predictors.
2. **Create the time splits.** Divide the record into eight blocked folds with
   separate training, validation and test periods.
3. **Build the model inputs.** Normalize each variable from the training split
   only. Encode the month or quarter as a seasonal signal.
4. **Train the model.** Encode Argo profiles with a graph neural network and
   the gridded fields with a multilayer perceptron. Combine both embeddings to
   reconstruct transport. Convert the final predictions back to Sverdrups for
   evaluation.
5. **Evaluate held-out periods.** Select the checkpoint with validation data.
   Use the test block only for the final metrics and predictions.

The pipeline supports monthly (`M`) and quarterly (`Q`) reconstruction. The
same method is used for both regimes.

</details>

<details>
<summary><strong>2. Directory structure</strong></summary>

<br>

| Path | Role | Main output |
| --- | --- | --- |
| [`dataset.py`](dataset.py) | Load the prepared data. Build blocked folds and training-only normalization. | PyTorch training, validation and test datasets. |
| [`model.py`](model.py) | Define the Argo graph encoder, surface encoder and prediction head. Validate checkpoint compatibility. | Transport predictions and latent embeddings. |
| [`train.py`](train.py) | Train one regime, fold, seed and spatial resolution. | `best.pt`, `history.csv` and `summary.json`. |
| [`evaluate.py`](evaluate.py) | Export predictions from one checkpoint and aggregate a complete campaign. | Prediction tables and reconstruction metrics. |
| [`initial_centers.npz`](initial_centers.npz) | Store the optional paper-reference graph initialization. | Initial graph compartment centers for each fold and regime. |
| [`analysis/`](analysis/README.md) | Run baselines, permutation tests and scientific diagnostics. | Analysis tables used by the paper figures. |
| `runs/` | Store generated training, evaluation and analysis outputs. | Local results. This directory is excluded from Git. |

</details>

<details>
<summary><strong>3. Choose a task</strong></summary>

Before either task, you can
[run a minimal end-to-end test](../README.md#minimal-end-to-end-test) to verify
the local installation without downloading the archived paper data.

### A. Evaluate archived models

Choose this route to reproduce predictions and metrics from the preserved
paper checkpoints. It belongs to
[Workflow A](../README.md#workflow-a) in the main README. Use the frozen data
and `base_models.zip` from the
[Zenodo record](https://doi.org/10.5281/zenodo.23287700).

Go to [Evaluate checkpoints](#evaluate-checkpoints).

### B. Train new models

Choose this route to create new checkpoints with the final method. It belongs
to [Workflow B](../README.md#workflow-b). Start from prepared data, then run one
or more regime, fold, seed and resolution combinations.

Go to [Train models](#train-models).

Scientific analyses can be run after either route when the required prediction
or checkpoint outputs are available.

</details>

<details>
<summary><strong>4. Requirements and prepared data</strong></summary>

Use Python 3.12 and install the pinned environment from the repository root:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

The default data location is `data_preparation/data/`. Use the frozen
`data.zip` archive or complete the source rebuild described in
[`data_preparation/README.md`](../data_preparation/README.md). Validate the
prepared data before training or inference:

```sh
python data_preparation/check_outputs.py
python -m unittest discover -s tests -t .
```

Run the remaining commands from the repository root.

</details>

<details>
<summary><strong>5. Dataset and model</strong></summary>

### Inputs and target

| Component | Model role |
| --- | --- |
| Virtual Argo density, position and relative time | Sparse input to the graph neural network. |
| SST, SSS, SSH and ocean bottom pressure | Gridded input to the surface encoder. |
| Month or quarter of the year | Seasonal input to the final representation. |
| Ekman-removed AMOC transport | Scalar reconstruction target in Sverdrups. |

### Time regimes

`M` uses one monthly target, one monthly surface row and the profiles from that
month. `Q` uses a three-month mean target, a three-month Argo profile cloud and
the mean of three consecutive monthly surface rows.

### Folds and normalization

Each regime is divided into eight contiguous test blocks. A separate adjacent
buffer with the same total duration is reserved for validation. The remaining
six blocks are used for training. Target, surface and Argo normalization
statistics are computed from the training samples only.

The supported surface resolutions are 20, 40, 80 and 160 longitude bins. They
use 20, 10, 5 and 3 latitude rows respectively.

</details>

<a id="train-models"></a>
<details>
<summary><strong>6. Train models</strong></summary>

The training command creates one run for a selected regime, fold, seed and
surface resolution.

### Step 1. Run one configuration

```sh
python -m machine_learning.train --regime M --fold 1 --seed 42 --bins 20
```

Use `M` or `Q`, folds 1 to 8, one of the 20 paper seeds and 20, 40, 80 or 160
bins. The default device is CPU. CUDA and MPS can be selected with `--device`.

### Step 2. Choose the graph initialization

The default `--center-init kmeans` computes graph compartment centers from the
current training split. Use the preserved initialization for a closer match to
the paper setup:

```sh
python -m machine_learning.train --regime M --fold 1 --seed 42 --bins 20 \
  --center-init historical
```

Both modes keep the centers trainable after initialization.

### Step 3. Read the run outputs

Each run is written to:

```text
machine_learning/runs/<regime>_<bins>/fold_<fold>/seed_<seed>/
```

The directory contains:

- `best.pt` with model weights, configuration and normalization statistics;
- `history.csv` with training and validation losses by epoch;
- `summary.json` with the selected epoch and final test metrics.

Existing run directories are not overwritten.

### Step 4. Scale to the full experiment

The base experiment contains 2 regimes, 8 folds and 20 seeds. This gives 320
runs at 20 bins. The spatial-resolution experiment repeats the design at 20,
40, 80 and 160 bins. This gives 1,280 runs.

Training uses batches of 16 and can run for up to 140 epochs. Early stopping
uses the validation block with a patience of 50 epochs. The selected checkpoint
balances validation loss and prediction spread. The test block is evaluated
only after selection.

</details>

<a id="evaluate-checkpoints"></a>
<details>
<summary><strong>7. Evaluate checkpoints</strong></summary>

Evaluation has two stages. First export the held-out predictions from each
checkpoint. Then aggregate the complete set of folds and seeds.

For archived inference, extract `base_models.zip` from the
[Zenodo record](https://doi.org/10.5281/zenodo.23287700) in the repository root
before running the commands below.

### Step 1. Export one checkpoint

```sh
python -m machine_learning.evaluate export \
  --checkpoint /path/to/best.pt \
  --regime M --bins 20 --fold 1 --seed 42 \
  --output-dir machine_learning/runs/evaluation/M_20/fold_01/seed_00042
```

The exporter checks the regime, fold, seed, resolution and normalization
contract before inference. It writes `predictions.csv` and `metrics.json`.

### Step 2. Export the complete campaign

Repeat the export command for every required fold and seed. Keep the same
directory pattern so each run remains identifiable.

### Step 3. Aggregate predictions

```sh
python -m machine_learning.evaluate aggregate \
  machine_learning/runs/evaluation/M_20/fold_*/seed_*/predictions.csv \
  machine_learning/runs/evaluation/Q_20/fold_*/seed_*/predictions.csv \
  --output-dir machine_learning/runs/evaluation_summary
```

Predictions are averaged across the 20 seeds for each test date. The six skill
metrics are then computed within each held-out fold. Final values are the mean
and sample standard deviation across the eight folds.

The aggregation writes:

- `seed_mean_predictions.csv` with one out-of-fold prediction per date;
- `fold_metrics.csv` with six metrics for each fold;
- `summary.csv` with the final mean and standard deviation for each regime.

</details>

<details>
<summary><strong>8. Analyse the results</strong></summary>

The [`analysis/`](analysis/README.md) directory contains four post-training
tasks:

| Task | Purpose |
| --- | --- |
| Baselines | Compare the neural model with Ridge regression and seasonal climatology. |
| Source permutation | Measure the change in error when one input source is permuted. |
| Regional permutation | Measure the change in error when one surface longitude bin is permuted. |
| Temporal and latent diagnostics | Examine recoverable time scales and the information retained by the learned representation. |

Read [`analysis/README.md`](analysis/README.md) for the required inputs and
commands for each task. The analyses create numerical tables. Read
[`figures/README.md`](../figures/README.md) to use those tables to regenerate
the paper figures.

</details>

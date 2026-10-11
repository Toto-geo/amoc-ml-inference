# Analysis

These scripts compare the trained models with simple baselines and examine
which signals contribute to the reconstructed transport.

<details>
<summary><strong>1. Overview</strong></summary>

The analysis follows the model evaluation stage.

1. Build Ridge and seasonal-climatology baselines from the same blocked time
   folds used by the neural model.
2. Permute complete input sources or individual longitude bins to measure
   their contribution to held-out predictions.
3. Separate slow and fast variability in the reconstructed series.
4. Test how much target information is retained in the learned latent
   representation.

Each script writes numerical tables below `machine_learning/runs/`. The
[`figures/`](../../figures/README.md) workflow turns the required tables into
paper figures.

</details>

<details>
<summary><strong>2. Directory structure</strong></summary>

<br>

| Script | Purpose | Main output |
| --- | --- | --- |
| [`baselines.py`](baselines.py) | Evaluate Ridge regression and seasonal climatology. | `baseline_predictions.csv` |
| [`permutation.py`](permutation.py) | Measure source and longitude-bin contributions. | `permutation_metrics.csv` and aggregated summaries |
| [`temporal.py`](temporal.py) | Decompose reconstructed series and evaluate their time scales. | Temporal components and band-skill tables |
| [`latent_knn.py`](latent_knn.py) | Compare model error with a nearest-neighbour estimate in latent space. | Per-seed and summary tables |

</details>

<details>
<summary><strong>3. Requirements and inputs</strong></summary>

Install the environment described in the main
[`README`](../../README.md#installation). Run all commands from the repository
root.

The required input depends on the analysis.

| Analysis | Required input |
| --- | --- |
| Baselines | Prepared paper data from `data.zip` or an equivalent completed data build. |
| Permutation | Prepared data and compatible model checkpoints. |
| Temporal | Complete seed-mean predictions from `machine_learning.evaluate aggregate`. |
| Latent k-NN | A complete monthly 20-bin campaign or previously exported embeddings and metadata. |

Use frozen data and archived checkpoints from the
[Zenodo record](https://doi.org/10.5281/zenodo.23287700) to analyse the paper
models. Read [`machine_learning/README.md`](../README.md) to prepare and
evaluate new runs.

Every command requires a new output directory. Existing results are not
overwritten.

</details>

<details>
<summary><strong>4. Run the baselines</strong></summary>

The baseline analysis uses the same eight blocked folds as the neural model.
Ridge regularisation is selected with the validation period. Feature scaling
and seasonal climatology are computed from the training period only.

Run both monthly and quarterly baselines:

```sh
python -m machine_learning.analysis.baselines --regime both \
  --output-dir machine_learning/runs/baselines
```

The command writes held-out Ridge and climatology predictions to
`baseline_predictions.csv`. This table is used with the model predictions in
Figure 6.

</details>

<details>
<summary><strong>5. Run the permutation tests</strong></summary>

Permutation tests measure how held-out error changes when model inputs are
shuffled in time.

- Source tests permute Argo or one complete surface field.
- Regional tests permute one longitude bin across all four surface fields.

Run one checkpoint:

```sh
python -m machine_learning.analysis.permutation run \
  --checkpoint /path/to/best.pt \
  --regime M --bins 20 --fold 1 --seed 42 \
  --bin-indices $(seq 1 20) \
  --output-dir machine_learning/runs/permutation/M_20/fold_01/seed_00042
```

Repeat the command for each required fold and seed. Then aggregate the full
monthly and quarterly campaign:

```sh
python -m machine_learning.analysis.permutation aggregate \
  machine_learning/runs/permutation/M_20/fold_*/seed_*/permutation_metrics.csv \
  machine_learning/runs/permutation/Q_20/fold_*/seed_*/permutation_metrics.csv \
  --output-dir machine_learning/runs/permutation_summary
```

The aggregate command checks that every analysis contains eight folds, 20
seeds and ten permutation repeats. Its summary table is used in Figures 7 and
8.

</details>

<details>
<summary><strong>6. Run the temporal and latent diagnostics</strong></summary>

### Temporal diagnostic

Use the complete seed-mean predictions created by model evaluation:

```sh
python -m machine_learning.analysis.temporal \
  --predictions machine_learning/runs/evaluation_summary/seed_mean_predictions.csv \
  --output-dir machine_learning/runs/temporal
```

The script separates the seasonal cycle, slow anomalies and faster residual
variability. It reports reconstruction skill for the slow and fast bands. The
decomposition is descriptive and is not used during training.

### Latent-space diagnostic

Use a complete monthly 20-bin campaign:

```sh
python -m machine_learning.analysis.latent_knn \
  --campaign-root machine_learning/runs \
  --output-dir machine_learning/runs/latent_knn
```

The script extracts the representation before the prediction head. It then
compares the model error with estimates from temporally separated neighbours
in each seed-specific latent space.

Read [`figures/README.md`](../../figures/README.md) to regenerate the paper
figures from the resulting tables.

</details>

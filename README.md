# AMOC ML Inference

**Reconstructing AMOC strength at 34.5°S from ocean reanalysis data using machine learning.**

This repository contains the data preparation, machine learning, evaluation
and figure code for the manuscript *Reconstructing AMOC Strength at 34.5°S
from Ocean Reanalysis Data Using Machine Learning* by Antonio Favaloro and
Cesar B. Rocha (DOI pending). Frozen data, trained models and numerical
results are stored in the matching
[Zenodo record](https://doi.org/10.5281/zenodo.23287700).

## Abstract

The Atlantic Meridional Overturning Circulation (AMOC) is a key component of
the climate system, redistributing heat, freshwater and water masses across
the Atlantic basin. Its strength is commonly described by the overturning
transport, which at 34.5°S is constrained by the SAMBA moored array.
Maintaining this array requires sustained logistical and financial effort,
motivating the search for alternative observing strategies. Here, we develop
a machine learning (ML) model to test whether Argo density profiles and
satellite-derived fields contain enough information to reconstruct the
overturning transport defined over the SAMBA interior section. The target and
synthetic observations are built within the ORAS5 ocean reanalysis, which
provides a long and physically consistent record for training the model at
monthly and quarterly scales. The analysis evaluates which part of the target
signal is recoverable and how it is constrained by the observations. The
results show that the model captures the target seasonal cycle and part of its
slow variability, while fast variability is largely missed. This recoverable
signal is mainly carried by subsurface information from Argo, with satellite
fields providing only a limited and localized contribution. These findings
point to autonomous subsurface sampling as the most promising direction for
future observing efforts, while showing that SAMBA remains difficult to
replace as a direct constraint on the overturning transport.

<details>
<summary><strong>1. Project overview</strong></summary>

The study tests whether observation-like inputs can reconstruct the Atlantic
Meridional Overturning Circulation at 34.5°S.

1. ORAS5 provides a continuous ocean record from 1958 to 2025. It is used to
   construct the Ekman-removed transport target and the model inputs.
2. Real Argo sampling patterns are applied to ORAS5 temperature and salinity
   to create virtual density profiles. ORAS5 also provides sea surface
   temperature, sea surface salinity, sea surface height and an ocean bottom
   pressure proxy.
3. A graph neural network encodes the sparse Argo profiles. A multilayer
   perceptron encodes the four gridded fields.
4. The two embeddings are combined with the month of the year. The model then
   predicts one transport value in Sverdrups.
5. Monthly and quarterly reconstructions are evaluated with eight blocked time
   folds and 20 random seeds. The analysis tests recoverable time scales,
   source contributions, regional contributions and surface resolution.

Parts of the methodological design were inspired by the work of
[Wölker et al. (2025)](https://doi.org/10.5194/os-21-3541-2025) and its
[source repository](https://github.com/ywoelker/amoc_reconstruction).

</details>

<details>
<summary><strong>2. Choose a workflow</strong></summary>

### Minimal end-to-end test

Verify the local installation before downloading the archived paper data. The
test generates small synthetic inputs and runs one complete training and
evaluation cycle on CPU.

Go to [Run a minimal end-to-end test](#minimal-end-to-end-test).

### Workflow A

Use the preserved data and results to inspect the paper workflow without
training new models. You can regenerate Figures 1-5, then Figures 6-9. You can
also run inference with the archived checkpoints.

Go to [Workflow A](#workflow-a).

### Workflow B

Use the final method to create new checkpoints. You can start from the frozen
paper data or rebuild the inputs from their sources. Each run produces one
model for a selected time regime, fold, seed and spatial resolution.

Go to [Workflow B](#workflow-b).

</details>

<details>
<summary><strong>3. Repository structure</strong></summary>

<br>

| Path | Purpose |
| --- | --- |
| [`data_preparation/`](data_preparation/README.md) | Download regional ORAS5 fields and Argo records. Apply selection and quality control. Construct the transport target, virtual Argo density profiles and four gridded model inputs. |
| [`machine_learning/`](machine_learning/README.md) | Load the prepared data, define the model, train checkpoints and evaluate predictions. |
| [`figures/`](figures/README.md) | Store the nine paper figures and one generator for each figure. |
| [`tests/`](tests/) | Check the data, model, evaluation and figure contracts. |
| [`examples/minimal_end_to_end/`](examples/minimal_end_to_end/README.md) | Generate synthetic inputs and run a compact training and evaluation test without archived data. |

The directories follow the main workflow: prepare the data, run the models,
generate the figures and verify the implementation. Detailed commands are in
the README of each component. Generated data and model runs are excluded from
Git.

</details>

<a id="installation"></a>
<details>
<summary><strong>4. Installation</strong></summary>

Python 3.12 is the tested version. From the repository root:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -t .
```

The pinned environment covers data preparation, training, evaluation and
figure generation.

### System requirements

The workflow has been tested on macOS and Linux with CPU execution. Training
also supports CUDA and Apple MPS through the `--device` option, but a GPU is
not required. A single 20-bin training run used about 0.7-0.75 GB of peak RAM
in the tested Linux CPU environment. Allow at least 2 GB of available RAM per
concurrent training worker. GPU memory has not been benchmarked, so no minimum
GPU-memory requirement is stated. Full experiments contain many independent
runs and require additional time and storage.

</details>

<a id="minimal-end-to-end-test"></a>
<details>
<summary><strong>5. Run a minimal end-to-end test</strong></summary>

Use this test to verify the installation before downloading the archived paper
data. It generates deterministic synthetic inputs and runs training and
evaluation for both time regimes.

```sh
python -m examples.minimal_end_to_end.generate_data
python -m examples.minimal_end_to_end.run
```

The test runs on CPU and does not reproduce scientific results. Read the
[`minimal end-to-end test README`](examples/minimal_end_to_end/README.md) for
the generated outputs and optional commands.

</details>

<a id="workflow-a"></a>
<details>
<summary><strong>6. Workflow A</strong></summary>

Use this workflow to reproduce the paper figures and archived predictions. It
does not train new models.

### Step 1. Install the environment

Complete the commands in [Installation](#installation).

### Step 2. Download and extract the artifacts

Download `data.zip` for Figures 1-5 and `results.zip` for Figures 6-9 from the
[Zenodo record](https://doi.org/10.5281/zenodo.23287700). The Zenodo README
explains how to reassemble and verify the split data archive. Extract the
reassembled archives from the repository root:

```sh
python -m zipfile -e /path/to/data.zip .
python -m zipfile -e /path/to/results.zip .
python data_preparation/check_outputs.py
```

### Step 3. Regenerate Figures 1-5

These figures use the frozen data products.

```sh
python -m figures.scripts.fig01_target --output-dir machine_learning/runs/fig01
python -m figures.scripts.fig02_argo_sampling --output-dir machine_learning/runs/fig02
python -m figures.scripts.fig03_sst_bins --output-dir machine_learning/runs/fig03
python -m figures.scripts.fig04_argo_gnn --output-dir machine_learning/runs/fig04
python -m figures.scripts.fig05_surface_encoder --output-dir machine_learning/runs/fig05
```

### Step 4. Regenerate Figures 6-9

These figures use the portable result tables in `results.zip`. They do not
require the model archives.

```sh
python -m figures.scripts.fig06_reconstruction \
  --predictions paper_results/predictions.csv \
  --baselines paper_results/baselines.csv \
  --output-dir machine_learning/runs/fig06
python -m figures.scripts.fig07_source_contribution \
  --permutation-summary paper_results/permutation_summary.csv \
  --output-dir machine_learning/runs/fig07
python -m figures.scripts.fig08_regional_contribution \
  --permutation-summary paper_results/permutation_summary.csv \
  --geometry paper_results/ssh_20bin_geometry.nc \
  --output-dir machine_learning/runs/fig08
python -m figures.scripts.fig09_spatial_resolution \
  --run-metrics paper_results/resolution_run_metrics.csv \
  --output-dir machine_learning/runs/fig09
```

### Step 5. Re-evaluate an archived model if needed

Download and extract `base_models.zip`, then follow the checkpoint export
example in [`machine_learning/README.md`](machine_learning/README.md). This
recreates predictions from preserved weights. It is separate from retraining.

(Generated PDFs can differ at the pixel or byte level across font and plotting
environments even when their numerical content agrees.)

</details>

<a id="workflow-b"></a>
<details>
<summary><strong>7. Workflow B</strong></summary>

Use this workflow to train and evaluate new models. It creates new checkpoints
with the final model and training method. Its output is a new set of runs that
can be evaluated and plotted.

### Step 1. Choose the data source

- Use `data.zip` from Zenodo for the closest comparison with the paper.
- Follow [`data_preparation/README.md`](data_preparation/README.md) to rebuild
  the inputs from ORAS5 and Argo services. Live sources can differ from the
  frozen paper data. A rebuild from current services is therefore a new data
  realization, not an exact reconstruction of the archived inputs.

### Step 2. Validate the prepared data

```sh
python data_preparation/check_outputs.py
```

### Step 3. Train one model

```sh
python -m machine_learning.train --regime M --fold 1 --seed 42 --bins 20
```

`M` is the monthly regime. `Q` is the quarterly regime. Training outputs are
written below `machine_learning/runs/`. Existing run directories are not
overwritten.

### Step 4. Scale to a complete experiment

The base experiment contains 2 regimes, 8 folds and 20 seeds. This gives 320
runs at 20 bins. The spatial-resolution experiment repeats the design at 20,
40, 80 and 160 bins. This gives 1,280 runs.

The final architecture and hyperparameters are fixed in the released code.
The historical optimization campaign used to select them is not included.

A new campaign follows the same method, but its weights and metrics need not
match the archived runs bit for bit across software and hardware environments.

### Step 5. Evaluate and analyse the runs

Use [`machine_learning/README.md`](machine_learning/README.md) to export and
aggregate predictions. Use
[`machine_learning/analysis/README.md`](machine_learning/analysis/README.md)
for baselines, permutation tests and diagnostics. The resulting tables can be
passed to the figure generators.

</details>

<details>
<summary><strong>8. Archived data and models</strong></summary>

The [Zenodo record](https://doi.org/10.5281/zenodo.23287700) separates the
artifacts by their role in the workflow.

| Archive | What it contains | Use it when |
| --- | --- | --- |
| `data.zip` | 34 frozen products, including the paper Argo catalog, target and model inputs | Regenerating Figures 1-5, running inference or retraining on the paper data |
| `results.zip` | Predictions, baselines, permutation results, resolution metrics and map geometry | Regenerating Figures 6-9 without model weights |
| `base_models.zip` | 320 monthly and quarterly 20-bin model runs | Recreating archived predictions and metrics from the preserved weights, without retraining |
| `figure09_models.zip` | 1,280 runs at 20, 40, 80 and 160 bins | Recomputing the Figure 9 resolution metrics from all preserved checkpoints instead of using the archived result table |

Zenodo stores `data.zip` as four transport-safe parts. Concatenating them
recreates the original archive. Checksums and member manifests are provided in
the record. Its included `README.md` provides the complete extraction and
verification instructions. Raw download caches and nonfinal experiments are
not included.

</details>

<details>
<summary><strong>9. Citation</strong></summary>

Cite the source code as:

> Favaloro, Antonio (2026). *AMOC ML Inference* (Version 1.0.0) [Computer
> software]. GitHub. https://github.com/Toto-geo/amoc-ml-inference/releases/tag/v1.0.0

Cite the archived data, checkpoints and numerical results as:

> Favaloro, Antonio (2026). *AMOC ML Inference: frozen data, model checkpoints
> and paper results* (Version 1.0.0) [Data set]. Zenodo.
> https://doi.org/10.5281/zenodo.23287700

When using the derived data, also cite the underlying ORAS5 and Argo records
listed below.

</details>

<details>
<summary><strong>10. License and data provenance</strong></summary>

The source code is available under the [BSD 3-Clause License](LICENSE). The
project-authored Zenodo artifacts are released under CC BY 4.0 to the extent
that the author holds rights in them.

The derived products use [ORAS5](https://doi.org/10.24381/cds.67e8eeb7) and
[Argo GDAC data](https://doi.org/10.17882/42182). Users must cite these sources
and follow their current terms and acknowledgement guidance.

</details>

<details>
<summary><strong>11. Contact</strong></summary>

Maintainer: [Antonio Favaloro](https://github.com/Toto-geo). Questions and
reproducibility reports can also be submitted through the GitHub issue tracker.

</details>

# Figures

This directory contains the nine paper figures and one Python generator for
each figure.

<details>
<summary><strong>1. Overview</strong></summary>

The figures follow the order of the paper.

- Figures 1-3 describe the transport target and model inputs.
- Figures 4-5 describe the two model encoders.
- Figures 6-9 present reconstruction skill, input contributions and spatial
  resolution.

The publication PDFs are stored in [`pdf/`](pdf/). The matching generators
are stored in [`scripts/`](scripts/). Generated files should be written below
`machine_learning/runs/` so the publication PDFs remain unchanged.

</details>

<details>
<summary><strong>2. Figure index</strong></summary>

<br>

| No. | Figure | PDF | Generator | Input |
| --- | --- | --- | --- | --- |
| 1 | AMOC target | [PDF](pdf/fig01_target.pdf) | [Python](scripts/fig01_target.py) | Prepared data |
| 2 | Argo sampling | [PDF](pdf/fig02_argo_sampling.pdf) | [Python](scripts/fig02_argo_sampling.py) | Prepared data |
| 3 | SST bin construction | [PDF](pdf/fig03_sst_bins.pdf) | [Python](scripts/fig03_sst_bins.py) | Prepared data |
| 4 | Argo-GNN encoder | [PDF](pdf/fig04_argo_gnn.pdf) | [Python](scripts/fig04_argo_gnn.py) | Prepared data |
| 5 | Surface encoder | [PDF](pdf/fig05_surface_encoder.pdf) | [Python](scripts/fig05_surface_encoder.py) | Code only |
| 6 | Reconstruction skill | [PDF](pdf/fig06_reconstruction.pdf) | [Python](scripts/fig06_reconstruction.py) | Predictions and baselines |
| 7 | Source contribution | [PDF](pdf/fig07_source_contribution.pdf) | [Python](scripts/fig07_source_contribution.py) | Permutation summary |
| 8 | Regional contribution | [PDF](pdf/fig08_regional_contribution.pdf) | [Python](scripts/fig08_regional_contribution.py) | Permutation summary and bin geometry |
| 9 | Spatial resolution | [PDF](pdf/fig09_spatial_resolution.pdf) | [Python](scripts/fig09_spatial_resolution.py) | Per-run metrics |

</details>

<details>
<summary><strong>3. Requirements and inputs</strong></summary>

Install the environment described in the main
[`README`](../README.md#installation). Run all commands from the repository
root.

For the shortest reproduction route, download and extract these archives from
the [Zenodo record](https://doi.org/10.5281/zenodo.23287700):

- `data.zip` provides the prepared inputs for Figures 1-4.
- `results.zip` provides the numerical tables for Figures 6-9.
- Figure 5 is generated directly from code.

New evaluation and analysis outputs can also be used. Read
[`machine_learning/README.md`](../machine_learning/README.md) and
[`machine_learning/analysis/README.md`](../machine_learning/analysis/README.md)
for the required steps.

Figure 8 uses Cartopy. Its Natural Earth coastline and land data must already
be available in the local Cartopy cache when working offline.

</details>

<details>
<summary><strong>4. Regenerate Figures 1-5</strong></summary>

Extract the prepared data in the repository root. Then run:

```sh
python -m figures.scripts.fig01_target \
  --output-dir machine_learning/runs/fig01
python -m figures.scripts.fig02_argo_sampling \
  --output-dir machine_learning/runs/fig02
python -m figures.scripts.fig03_sst_bins \
  --output-dir machine_learning/runs/fig03
python -m figures.scripts.fig04_argo_gnn \
  --output-dir machine_learning/runs/fig04
python -m figures.scripts.fig05_surface_encoder \
  --output-dir machine_learning/runs/fig05
```

Each command writes one PDF to its selected output directory.

</details>

<details>
<summary><strong>5. Regenerate Figures 6-9</strong></summary>

Extract `results.zip` in the repository root. Then run:

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

Figures 6-8 also write the numerical rows used in each plot. Figure 9 can read
a complete model campaign with `--runs-root` instead of the archived metric
table.

PDF rendering can vary slightly across font and plotting-library versions.
The numerical plot data should remain unchanged.

</details>

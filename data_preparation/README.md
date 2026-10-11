# Data preparation

This directory builds the AMOC target and the observation-like inputs used by
the machine learning workflow.

<details>
<summary><strong>1. Overview</strong></summary>

The pipeline prepares the complete 1958-2025 dataset used by the model at
34.5°S.

1. **Collect the source data.** Extract the required ORAS5 fields and obtain
   the Argo profiles that define the sampling pattern.
2. **Construct the scientific products.** Build the Ekman-removed transport
   target, the ocean bottom pressure proxy and the virtual Argo density
   profiles. Aggregate SST, SSS, SSH and ocean bottom pressure into the model
   geometries.
3. **Write the model inputs.** Save monthly and quarterly targets, monthly and
   quarterly Argo inputs, and the four gridded predictors at each supported
   spatial resolution.

Zonal wind stress is required to construct the target. It is not a model
predictor.

</details>

<details>
<summary><strong>2. Directory structure</strong></summary>

<br>

| Path | Role | Main output |
| --- | --- | --- |
| [`dataset_construction/source/`](dataset_construction/source/) | Collect and subset the seven regional ORAS5 fields. | Monthly temperature, salinity, SST, SSS, SSH, zonal wind stress and meridional velocity. |
| [`dataset_construction/derived/`](dataset_construction/derived/) | Derive physical quantities that are not direct ORAS5 outputs. | Ocean bottom pressure and Ekman velocity. |
| [`dataset_construction/derived/argo/`](dataset_construction/derived/argo/) | Download candidate Argo profiles, apply quality control and build the virtual sampling geometry. | Frozen profile catalog and virtual profile locations. |
| [`target_construction/`](target_construction/) | Convert meridional and Ekman velocity into the reconstruction target. | Monthly and quarterly AMOC transport in Sverdrups. |
| [`input_construction/`](input_construction/) | Sample density at virtual Argo locations and bin the four gridded fields. | Model-ready Argo and surface inputs. |
| [`check_outputs.py`](check_outputs.py) | Validate the final data contract. | Pass or fail report for the model and figure inputs. |
| `data/` | Store downloads, intermediate products and final outputs. | Local generated data. This directory is excluded from Git. |

</details>

<details>
<summary><strong>3. Choose a starting point</strong></summary>

### A. Use the frozen paper data

Choose this route for [Workflow A](../README.md#workflow-a) in the main README.
It reproduces the paper figures and archived inference without rebuilding the
dataset. It is also the recommended starting point for
[Workflow B](../README.md#workflow-b) when the aim is to train on the same
inputs used by the paper. Download `data.zip` from the
[Zenodo record](https://doi.org/10.5281/zenodo.23287700).

Go to [Use the frozen data](#frozen-data).

### B. Rebuild the data from source

Choose this route as the data-construction stage of
[Workflow B](../README.md#workflow-b). It is intended for inspecting the
complete pipeline or training on a new dataset built from the currently
available ORAS5 and Argo services. Live sources can differ from the frozen
paper data.

Go to [Rebuild from source](#rebuild-from-source).

</details>

<details>
<summary><strong>4. Requirements</strong></summary>

Use Python 3.12 and install the pinned environment from the repository root:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

A full rebuild also requires:

- access to the historical ORAS5 ICDC OPeNDAP archive;
- a configured Copernicus Climate Data Store account for recent ORAS5 files;
- acceptance of the relevant CDS dataset terms;
- access to the Argo GDAC profile archive;
- enough disk space for about 5 GB of retained data and additional raw caches.

Run every command from the repository root. No scientific sample dataset is
bundled with the source code. The
[minimal end-to-end test](../README.md#minimal-end-to-end-test) generates
separate synthetic inputs and does not exercise data preparation.

</details>

<a id="frozen-data"></a>
<details>
<summary><strong>5. Use the frozen data</strong></summary>

The [Zenodo record](https://doi.org/10.5281/zenodo.23287700) stores the large
data archive in four transport-safe parts. Follow its `README.md` to verify and
join the parts. Then extract the reassembled archive from the repository root:

```sh
python -m zipfile -e /path/to/data.zip .
python data_preparation/check_outputs.py
```

The archive already contains the correct `data_preparation/data/` paths. A
successful check confirms that the model and figure input contracts are
complete.

</details>

<a id="rebuild-from-source"></a>
<details>
<summary><strong>6. Rebuild from source</strong></summary>

Use a fresh `data_preparation/data/` directory. The builders protect existing
outputs and normally stop instead of replacing them.

Define the paths used below:

```sh
DATA=data_preparation/data
SRC=data_preparation/dataset_construction/source
DER=data_preparation/dataset_construction/derived
ARGO="$DER/argo"
TC=data_preparation/target_construction
IC=data_preparation/input_construction
RAW="$DATA/raw"
```

### Step 1. Build the regional ORAS5 fields

These commands combine the historical archive with recent CDS files. Raw
recent files are kept below `data/raw/` and can be removed after a successful
build.

```sh
python "$SRC/build_sst.py" --recent-raw-dir "$RAW/sst" --download-recent --output-dir "$DATA/dynamic/SST"
python "$SRC/build_sss.py" --recent-raw-dir "$RAW/sss" --download-recent --output-dir "$DATA/dynamic/SSS"
python "$SRC/build_ssh.py" --recent-raw-dir "$RAW/ssh" --download-recent --output-dir "$DATA/dynamic/SSH"
python "$SRC/build_temperature.py" --recent-raw-dir "$RAW/temperature" --download-recent --output-dir "$DATA/dynamic/Temperature"
python "$SRC/build_salinity.py" --recent-raw-dir "$RAW/salinity" --download-recent --output-dir "$DATA/dynamic/Salinity"
python "$SRC/build_wind_stress.py" --recent-raw-dir "$RAW/wind_stress" --download-recent --output-dir "$DATA/dynamic/zonal_wind_stress"
python "$SRC/build_meridional_velocity.py" --recent-raw-dir "$RAW/meridional_velocity" --download-recent --output-dir "$DATA/dynamic/meridional_velocity"
```

Define the resulting source files:

```sh
T="$DATA/dynamic/Temperature/oras5_votemper_34p5S_pm3deg_1958_2025.nc"
S="$DATA/dynamic/Salinity/oras5_vosaline_34p5S_pm3deg_1958_2025.nc"
H="$DATA/dynamic/SSH/oras5_sossheig_34p5S_pm2deg_1958_2025.nc"
V="$DATA/dynamic/meridional_velocity/oras5_vomecrty_34p5S_1958_2025.nc"
W="$DATA/dynamic/zonal_wind_stress/oras5_sozotaux_34p5S_pm1p5deg_1958_2025.nc"
MESH="$DATA/static/mesh_mask.nc"
```

### Step 2. Build the derived fields and transport target

Ocean bottom pressure is derived from temperature, salinity and sea surface
height. Ekman velocity is derived from meridional velocity and zonal wind
stress. The target builder then produces monthly and quarterly transport.

```sh
python "$DER/build_obp.py" --temperature "$T" --salinity "$S" --ssh "$H" --mesh-mask "$MESH" --download-mesh --output-dir "$DATA/dynamic/OBP"
python "$DER/build_ekman_velocity.py" --velocity "$V" --wind-stress "$W" --mesh-mask "$MESH" --output-dir "$DATA/dynamic/ekman_velocity"
python "$TC/build_target.py" --velocity "$V" --ekman "$DATA/dynamic/ekman_velocity/oras5_ekman_velocity_34p5S_1958_2025.nc" --output-dir "$DATA/target"
```

### Step 3. Build the virtual Argo geometry

The frozen candidate manifest identifies the source profiles. The pipeline
downloads them, applies quality control and extends the selected sampling
pattern across the ORAS5 period.

```sh
python "$ARGO/build_argo_all.py" --candidates "$ARGO/argo_candidates_2003_2025.csv" --cache-dir "$DATA/argo/cache" --output "$DATA/argo/01_argo_all.csv"
python "$ARGO/build_argo_qc.py" --all-profiles "$DATA/argo/01_argo_all.csv" --output "$DATA/argo/02_argo_QC.csv"
python "$ARGO/build_argo_geometry.py" --qc-csv "$DATA/argo/02_argo_QC.csv" --temperature "$T" --mesh-mask "$MESH" --output-dir "$DATA/argo"
```

### Step 4. Build the model inputs

The Argo builder samples ORAS5 density at the virtual profile locations. The
surface builder creates SST, SSS, SSH and ocean bottom pressure inputs at all
four spatial resolutions.

```sh
python "$IC/build_argo_density.py" --geometry "$DATA/argo/oras5_virtual_argo_coverage_geometry_34p5S_pm3deg_1958_2025.parquet" --qc-csv "$DATA/argo/02_argo_QC.csv" --temperature "$T" --salinity "$S" --output-dir "$DATA/input/virtual_argo"
python "$IC/build_surface_inputs.py"
```

### Step 5. Validate the outputs

```sh
python data_preparation/check_outputs.py
```

</details>

<details>
<summary><strong>7. Output contract</strong></summary>

The final model-facing dataset contains:

- two transport targets, one monthly and one quarterly;
- two virtual Argo density inputs, one monthly and one quarterly;
- four monthly gridded predictors at four spatial resolutions, for 16 surface
  input files.

The supported surface geometries are 20 bins by 20 latitude rows, 40 by 10,
80 by 5 and 160 by 3. Surface files contain physical bin means and geometry.
They do not contain global normalization statistics. Normalization is computed
from each training fold inside the machine learning pipeline.

For the quarterly model, the target is a non-overlapping three-month mean and
the Argo input contains the three-month profile cloud. Surface fields remain
monthly files. Their three-month means are computed by the machine learning
loader before fold-specific normalization.

</details>

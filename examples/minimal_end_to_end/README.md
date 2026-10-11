# Minimal end-to-end test

This example tests the machine-learning pipeline without downloading the paper
data. It uses small deterministic synthetic inputs.

<details>
<summary><strong>1. Purpose</strong></summary>

Use this workflow to check that the local environment can load the data, build
the model, train one epoch, save a checkpoint and evaluate it.

This is a technical smoke test. Its inputs, predictions and metrics are not
scientific results and cannot reproduce the paper figures.

</details>

<details>
<summary><strong>2. Run the test</strong></summary>

Install the environment described in the main
[`README`](../../README.md#installation). Run these commands from the
repository root:

```sh
python -m examples.minimal_end_to_end.generate_data
python -m examples.minimal_end_to_end.run
```

The first command generates about 2.4 MB of synthetic monthly and quarterly
inputs. The second command runs both regimes on CPU. Each regime uses fold 1,
seed 42, 20 surface bins and one training epoch.

</details>

<details>
<summary><strong>3. Run one regime</strong></summary>

Use `M` for monthly data or `Q` for quarterly data:

```sh
python -m examples.minimal_end_to_end.run --regime M
python -m examples.minimal_end_to_end.run --regime Q
```

Select new directories with `--data-root` and `--output-dir`. Existing run
directories are not overwritten.

</details>

<details>
<summary><strong>4. Outputs</strong></summary>

Generated data and runs are written below
`examples/minimal_end_to_end/generated/`. This directory is excluded from Git.

Each regime creates a checkpoint, training history, summary, predictions and
evaluation metrics. Successful completion verifies the software path only.

</details>

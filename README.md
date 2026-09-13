# JetFighter Evaluation

Human validation of the JetFighter pipeline. The full instructions are in
**`JetFighter Evaluation Protocol.pdf`**. This README only covers how to run it.

Each dataset is a folder containing a `manifest.csv` and its image files. The dataset name is selected at runtime with the `--dataset` flag. Each reviewer runs their own copy of this repo and receives a separate CSV per dataset in
`results/`.

## Docker

```bash
docker compose build
docker compose up ui                  # default dataset at http://localhost:8501
docker compose run --rm ui metrics    # metrics for the default dataset
docker compose run --rm ui report     # report for the default dataset
```

To select another dataset in Docker, pass the flag after the service name. Use
`--service-ports` when launching the UI with `run`:

```bash
docker compose run --rm --service-ports ui ui --port 8501 --dataset my_dataset
docker compose run --rm ui metrics --dataset my_dataset
docker compose run --rm ui report --dataset my_dataset
```

Inside the container, this is equivalent to
`python -m evaluation ui --dataset <dataset-folder>`. The folder name is resolved below `/app/evaluation/`; do not use a Windows host path there.

## Local (no Docker)

```bash
pip install -r requirements.txt

python -m evaluation ui --dataset dataset   # http://localhost:8501
```

After the human review is complete, add the corresponding `predictions.csv` to your local dataset copy, then run:

```bash
python -m evaluation metrics --dataset <dataset-folder>
python -m evaluation report --dataset <dataset-folder>
```

## Resume

Labels auto-save after every submit. To continue where you left off, just start the UI again —> it jumps to the next unlabelled crop:

```bash
python -m evaluation ui --dataset <dataset-folder>
```

Outputs use the selected dataset name, for example
`ground_truth_<dataset-folder>.csv` and `metrics_<dataset-folder>.json`.

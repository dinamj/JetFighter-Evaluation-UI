# JetFighter Evaluation

Human validation of the JetFighter pipeline. The full instructions are in
**`JetFighter Evaluation Protocol.pdf`**. This README only covers how to run it.

The dataset (crops + pipeline predictions) ships in `dataset/`. Each reviewer
runs their own copy of this repo, labels every crop, and ends up with their own
`results/ground_truth.csv`.

## Docker

```bash
docker compose build
docker compose up ui                  # label crops at http://localhost:8501
docker compose run --rm ui metrics    # pipeline vs your labels
docker compose run --rm ui report     # results/report.html
```

## Local (no Docker)

```bash
pip install -r requirements.txt

python -m evaluation ui                 # http://localhost:8501
python -m evaluation metrics
python -m evaluation report
```

## Resume

Labels auto-save after every submit. To continue where you left off, just start
the UI again —> it jumps to the next unlabelled crop:

```bash
python -m evaluation ui
```

Outputs land in `results/`: `ground_truth.csv` (your labels), `metrics.json`,
`metrics_report.txt`, `report.html`.

# Breast Cancer MLOps Pipeline

Train a Random Forest classifier on the Breast Cancer Wisconsin Diagnostic dataset and serve it with FastAPI. The model predicts whether a tumor is benign or malignant from 30 numeric measurements.

## Dataset

The training CSV is `data/raw/data.csv`. The `diagnosis` column is the target (`B` = benign, `M` = malignant); the unique `id` column is excluded from training. The supplied dataset has 569 rows and no missing values.

To use another copy of the Kaggle dataset, place its CSV at `data/raw/data.csv`. It must contain `id`, `diagnosis`, and the 30 numeric feature columns in the Breast Cancer Wisconsin Diagnostic dataset.

## Run locally

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
python -m src.train
uvicorn app.main:app --reload
```

Open the interactive API docs at http://127.0.0.1:8000/docs. `POST /predict` accepts all 30 measurement fields and returns `benign` or `malignant` with a confidence score. `GET /model-info` reports the training metadata.

## Tests and lint

```bash
pytest -v
ruff check .
```

## Docker

```bash
docker build -t breast-cancer-mlops .
docker run -p 8000:8000 breast-cancer-mlops
```

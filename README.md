# ⚽ Footballer Value ML

![CI Pipeline](https://github.com/CarlGvM/footballer-value-ml/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.13-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2-orange)
![Docker](https://img.shields.io/badge/Docker-ready-blue)

A full end-to-end **MLOps project** that predicts football player market values using machine learning. Built to demonstrate DevOps and MLOps best practices including data pipelines, experiment tracking, model serving, containerisation, and CI/CD automation.

> *"How much is a footballer actually worth?"* — This project answers that question with a trained XGBoost model achieving **R² = 0.9983** on real FIFA 22 player data.

---

## 🖥️ Demo

![Footballer Value Predictor UI](docs/screenshot.png)

A dark, football-themed web interface where you can adjust player attributes using sliders and instantly get an estimated market value in euros.

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Data Pipeline │────▶│   ML Training    │────▶│   Model Store   │
│  (FIFA 22 CSV)  │     │ (XGBoost+MLflow) │     │  (.pkl + runs)  │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                           │
                                                           ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│    Frontend     │────▶│   FastAPI REST   │────▶│  Trained Model  │
│  (HTML/CSS/JS)  │     │      API         │     │  (Prediction)   │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                │
                                ▼
                    ┌──────────────────────┐
                    │   Docker Container   │
                    │  GitHub Actions CI   │
                    └──────────────────────┘
```

---

## 🚀 Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.13 |
| ML Model | XGBoost |
| Experiment Tracking | MLflow |
| API | FastAPI + Uvicorn |
| Frontend | HTML / CSS / JavaScript |
| Containerisation | Docker |
| CI/CD | GitHub Actions |
| Data | FIFA 22 Complete Player Dataset (Kaggle) |

---

## 📁 Project Structure

```
footballer-value-ml/
├── .github/
│   └── workflows/
│       └── ci.yml              # CI/CD pipeline
├── data/
│   ├── raw/                    # Original FIFA 22 CSV
│   └── processed/              # Cleaned, feature-engineered data
├── models/
│   └── player_value_model.pkl  # Serialised trained model
├── src/
│   ├── api/
│   │   └── main.py             # FastAPI app
│   ├── data/
│   │   ├── fetch_data.py       # Data extraction
│   │   └── process_data.py     # Data processing pipeline
│   ├── frontend/
│   │   └── index.html          # Web UI
│   └── models/
│       └── train.py            # Model training with MLflow
├── tests/
│   └── test_api.py             # API unit tests
├── conftest.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🧠 ML Pipeline

### Data
- **Source:** FIFA 22 Complete Player Dataset (19,239 players, 110 features)
- **Processing:** Feature selection, position grouping (ATT/MID/DEF/GK), work rate encoding, one-hot encoding, log transformation of target variable (`np.log1p`)
- **Final dataset:** 17,041 rows × 26 columns

### Features Used
`age`, `overall`, `potential`, `wage_eur`, `international_reputation`, `weak_foot`, `skill_moves`, `pace`, `shooting`, `passing`, `dribbling`, `defending`, `physic`, `preferred_foot`, `years_on_contract`, `position`, `attack_work_rate`, `defense_work_rate`

### Model Performance

| Metric | Value |
|---|---|
| R² Score | **0.9983** |
| RMSE (log scale) | 0.0497 |
| MAE (log scale) | 0.0342 |

### Key Design Decisions
- **Log transformation** on `value_eur` — real player values are heavily right-skewed (most players worth <€1M, a few worth €100M+). Log scaling helps the model learn patterns across the full value range
- **XGBoost** chosen for its strong performance on tabular data and interpretability
- **MLflow** tracks every experiment run — parameters, metrics, and model artifacts are logged automatically
- **Data quality validation** — an initial synthetic dataset was identified and discarded after near-zero feature correlations with the target variable confirmed it was artificially generated

---

## 🔌 API Reference

The prediction API is built with FastAPI and auto-generates interactive documentation at `/docs`.

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/health` | Service status |
| POST | `/predict` | Predict player market value |

### Example Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 22,
    "overall": 82,
    "potential": 91,
    "wage_eur": 25000,
    "international_reputation": 2,
    "weak_foot": 3,
    "skill_moves": 4,
    "pace": 85,
    "shooting": 78,
    "passing": 80,
    "dribbling": 86,
    "defending": 32,
    "physic": 68,
    "preferred_foot": 1,
    "years_on_contract": 3,
    "position_ATT": 1,
    "position_DEF": 0,
    "position_GK": 0,
    "position_MID": 0,
    "attack_work_rate_High": 1,
    "attack_work_rate_Low": 0,
    "attack_work_rate_Medium": 0,
    "defense_work_rate_High": 0,
    "defense_work_rate_Low": 0,
    "defense_work_rate_Medium": 1
  }'
```

### Example Response

```json
{
  "predicted_value_eur": 76438464.0,
  "predicted_value_millions": 76.44
}
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.13+
- Docker (or Podman)
- Git

### Local Development

```bash
# Clone the repository
git clone https://github.com/CarlGvM/footballer-value-ml.git
cd footballer-value-ml

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download the dataset (requires Kaggle API token)
kaggle datasets download -d stefanoleone992/fifa-22-complete-player-dataset -p data/raw/ --unzip

# Process data
python src/data/process_data.py

# Train model
python src/models/train.py

# Start the API
uvicorn src.api.main:app --reload
```

Then open `src/frontend/index.html` in your browser.

### MLflow UI

```bash
mlflow ui
```

Navigate to `http://127.0.0.1:5000` to view experiment runs.

### Docker

```bash
# Build the image
docker build -t footballer-value-ml .

# Run the container
docker run -p 8000:8000 footballer-value-ml
```

---

## 🔄 CI/CD Pipeline

Every push to `main` automatically triggers:

1. **Lint** — `ruff` checks code quality across `src/`
2. **Test** — `pytest` runs all unit tests
3. **Build** — Docker image is built to confirm containerisation works

The pipeline is defined in `.github/workflows/ci.yml`.

---

## ⚠️ Limitations & Future Work

- **Data:** FIFA ratings are subjective estimates, not real scouting data. Injury history and disciplinary records — which significantly affect real transfer values — are not available in this dataset
- **Temporal drift:** The model is trained on FIFA 22 data (2022). Player values change season to season and the model would need periodic retraining
- **Deployment:** AWS cloud deployment planned for a future phase
- **Multi-season training:** Seasons FIFA 15–21 are available and could be incorporated for richer training data

---

## 👤 Author

**Carl** — [@CarlGvM](https://github.com/CarlGvM)

*Built with the assistance of Claude (Anthropic)*

---

## 📄 License

MIT License — feel free to use this project as a reference or starting point.
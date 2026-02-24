import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
import mlflow.xgboost
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

PROCESSED_DATA_PATH = Path("data/processed/players_processed.csv")

def train_model():
    """Train XGBoost model to predict player market value."""

    print("Loading processed data...")
    df = pd.read_csv(PROCESSED_DATA_PATH)

    # Split features and target
    X = df.drop(columns=["value_eur"])
    y = df["value_eur"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    # MLflow experiment
    mlflow.set_experiment("footballer-value-prediction")

    with mlflow.start_run():
        params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "random_state": 42
        }

        print("Training XGBoost model...")
        model = XGBRegressor(**params)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Convert log metrics back to euros for readability
        rmse_eur = np.expm1(rmse)
        mae_eur = np.expm1(mae)

        print(f"RMSE: {rmse:.4f} (log scale) | €{rmse_eur:,.0f}")
        print(f"MAE:  {mae:.4f} (log scale) | €{mae_eur:,.0f}")
        print(f"R²:   {r2:.4f}")

        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.xgboost.log_model(model, "model")

        print("Run logged to MLflow!")

if __name__ == "__main__":
    train_model()
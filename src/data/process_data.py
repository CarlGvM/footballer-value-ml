import pandas as pd
from pathlib import Path

RAW_DATA_PATH = Path("data/raw/fifa_player_performance_market_value.csv")
PROCESSED_DATA_PATH = Path("data/processed")
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

def process_data() -> pd.DataFrame:
    """
    Clean and prepare raw FIFA player data for model training.
    Returns processed DataFrame.
    """
    print("Loading raw data...")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Raw data shape: {df.shape}")

    # Drop columns not useful for prediction
    df = df.drop(columns=["player_id", "player_name", "nationality", "club"])

    # Encode binary categorical column
    df["injury_prone"] = df["injury_prone"].map({"Yes": 1, "No": 0})

    # Encode ordinal categorical column
    risk_map = {"Low": 0, "Medium": 1, "High": 2}
    df["transfer_risk_level"] = df["transfer_risk_level"].map(risk_map)

    # One-hot encode position
    df = pd.get_dummies(df, columns=["position"])

    # Drop any rows with missing values
    df = df.dropna()

    print(f"Processed data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Save
    output_path = PROCESSED_DATA_PATH / "players_processed.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    return df

if __name__ == "__main__":
    process_data()
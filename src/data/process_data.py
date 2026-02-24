import pandas as pd
import numpy as np
from pathlib import Path

RAW_DATA_PATH = Path("data/raw/players_22.csv")
PROCESSED_DATA_PATH = Path("data/processed")
PROCESSED_DATA_PATH.mkdir(parents=True, exist_ok=True)

def process_data() -> pd.DataFrame:
    """
    Clean and prepare FIFA 22 player data for model training.
    Returns processed DataFrame.
    """
    print("Loading raw data...")
    df = pd.read_csv(RAW_DATA_PATH, low_memory=False)
    print(f"Raw data shape: {df.shape}")

    # Drop rows where target variable is missing
    df = df.dropna(subset=["value_eur"])

    # Select relevant features + target
    features = [
        "age", "overall", "potential", "wage_eur",
        "international_reputation", "weak_foot", "skill_moves",
        "pace", "shooting", "passing", "dribbling", "defending", "physic",
        "preferred_foot", "work_rate", "player_positions",
        "club_contract_valid_until", "value_eur"
    ]
    df = df[features]

    # Engineer contract years remaining
    FIFA_22_YEAR = 2022
    df["years_on_contract"] = df["club_contract_valid_until"] - FIFA_22_YEAR
    df = df.drop(columns=["club_contract_valid_until"])

    # Simplify positions to primary position only (e.g. "ST, CF" -> "ST")
    df["position"] = df["player_positions"].str.split(",").str[0].str.strip()
    df = df.drop(columns=["player_positions"])

    # Group positions into broader categories
    position_map = {
        "GK": "GK",
        "CB": "DEF", "LB": "DEF", "RB": "DEF", "LWB": "DEF", "RWB": "DEF",
        "CDM": "MID", "CM": "MID", "CAM": "MID", "LM": "MID", "RM": "MID",
        "LW": "ATT", "RW": "ATT", "ST": "ATT", "CF": "ATT", "RF": "ATT", "LF": "ATT"
    }
    df["position"] = df["position"].map(position_map)

    # Encode preferred foot
    df["preferred_foot"] = df["preferred_foot"].map({"Right": 1, "Left": 0})

    # Simplify work rate (e.g. "High/Medium" -> two separate columns)
    df["attack_work_rate"] = df["work_rate"].str.split("/").str[0].str.strip()
    df["defense_work_rate"] = df["work_rate"].str.split("/").str[1].str.strip()
    df = df.drop(columns=["work_rate"])

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=["position", "attack_work_rate", "defense_work_rate"])

    # Drop remaining rows with missing values
    df = df.dropna()

    # Log transform the target variable to handle skew
    df["value_eur"] = np.log1p(df["value_eur"])

    print(f"Processed data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Save
    output_path = PROCESSED_DATA_PATH / "players_processed.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    return df

if __name__ == "__main__":
    process_data()
import soccerdata as sd
import pandas as pd
from pathlib import Path

# Define paths
RAW_DATA_PATH = Path("data/raw")
RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)

def fetch_player_valuations(leagues: list, seasons: list) -> pd.DataFrame:
    """
    Fetch player market valuations from Transfermarkt.
    
    Args:
        leagues: list of league codes e.g. ["ENG-Premier League"]
        seasons: list of seasons e.g. ["2223", "2324"]
    
    Returns:
        DataFrame with player valuations
    """
    print(f"Fetching valuations for {leagues}, seasons {seasons}...")
    
    tm = sd.Transfermarkt(leagues=leagues, seasons=seasons)
    df = tm.read_player_market_values()
    
    # Save raw data
    output_path = RAW_DATA_PATH / "player_valuations.csv"
    df.to_csv(output_path)
    print(f"Saved {len(df)} records to {output_path}")
    
    return df

if __name__ == "__main__":
    df = fetch_player_valuations(
        leagues=[
            "ENG-Premier League",
            "ESP-La Liga",
            "GER-Bundesliga",
            "ITA-Serie A",
            "FRA-Ligue 1",
            "NED-Eredivisie",
            "POR-Liga NOS",
            "BEL-First Division A",
            "TUR-Süper Lig",
        ],
        seasons=["2122", "2223", "2324"]
    )
    print(df.head())
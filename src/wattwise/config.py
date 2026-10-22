from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = ROOT / "data" / "raw" / "energydata_complete.csv"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
UCI_URL = (
    "https://archive.ics.uci.edu/static/public/374/appliances+energy+prediction.zip"
)
SEED = 42

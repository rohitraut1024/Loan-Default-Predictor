from pathlib import Path
import pandas as pd
import tomllib  # Python 3.11+; if 3.10, use 'tomli' package

def load_settings(cfg_path: str = "configs/settings.toml") -> dict:
    with open(cfg_path, "rb") as f:
        return tomllib.load(f)

def get_raw_csv_path(settings: dict) -> Path:
    raw_dir = Path(settings["paths"]["raw_dir"])
    fname = settings["data"]["raw_filename"]
    return raw_dir / fname

def quick_head(n: int = 5):
    """
    Load the raw CSV and return head() to verify everything is wired up.
    """
    settings = load_settings()
    csv_path = get_raw_csv_path(settings)
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find raw CSV at: {csv_path}")
    df = pd.read_csv(csv_path)
    return df.head(n)

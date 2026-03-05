import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
SRC_DIR = BASE_DIR / "src"
MODELS_DIR = Path(os.getenv("MODEL_DIR_PATH", str(BASE_DIR / "models")))
WEB_DIR = SRC_DIR / "web"
DATA_DIR = BASE_DIR / "data"

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

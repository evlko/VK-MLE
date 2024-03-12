import os
from pathlib import Path

from dotenv import load_dotenv

dotenv_path = Path("../.env")
load_dotenv()

ML_FLOW_URL = os.getenv("ML_FLOW_URL")
PROD = False
PROD_MODEL_NAME = "VK_MODEL"
PROD_MODEL_VERSION = 1
PROD_MODEL_PATH = f"{ML_FLOW_URL}:5000/models:/{PROD_MODEL_NAME}/{PROD_MODEL_VERSION}"
MODEL_PATH = "app/model.pt"
FEATURES_INFO_PATH = "app/scaler_info.json"

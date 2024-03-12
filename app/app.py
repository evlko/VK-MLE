import json
from typing import List

import mlflow
import torch
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.models import OpenAPI
from fastapi.openapi.utils import get_openapi
from starlette.responses import JSONResponse

import app.config as config
from app.schema import FeaturesData

app = FastAPI()


def load_model():
    if config.PROD:
        return mlflow.pytorch.load_model(config.PROD_MODEL_PATH)
    return torch.jit.load(config.MODEL_PATH)


def load_features_info():
    with open(config.FEATURES_INFO_PATH, "r") as file:
        scaler_info = json.load(file)
    return scaler_info


model = load_model()
model.eval()
features_info = load_features_info()


@app.get("/health")
def get_server_status():
    """Check API status"""
    status = {"health_check": "OK"}
    return JSONResponse(content=jsonable_encoder(status))


@app.post("/prediction")
def get_prediction(objects: List[FeaturesData]) -> int:
    """Get prediction from loaded model based on"""
    scores = []

    for object in objects:
        features = object.features
        values = []
        for feature in features_info["max"]:
            if feature in features:
                values.append(
                    (features[feature] - features_info["min"][feature])
                    / (features_info["max"][feature] - features_info["min"][feature])
                )
        input_tensor = torch.tensor([values], dtype=torch.float32)
        try:
            with torch.no_grad():
                score = model(input_tensor)
                scores.append(int(torch.squeeze(score)))
        except RuntimeError as e:
            scores.append(-1)

    result = {"predictions": scores}

    return JSONResponse(content=jsonable_encoder(result))


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(openapi_url="/openapi.json", title="docs")


@app.get("/openapi.json", include_in_schema=False)
async def get_open_api_endpoint():
    return JSONResponse(get_openapi(title="docs", version="0.1.0", routes=app.routes))

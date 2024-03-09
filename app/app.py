from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from starlette.responses import JSONResponse
from app.schema import FeaturesData
import torch

app = FastAPI()


def load_model():
    return torch.jit.load("app/model.pt")


model = load_model()
model.eval()


@app.get("/health")
def get_server_status():
    """Check API status"""
    status = {"health_check": "OK"}
    return JSONResponse(content=jsonable_encoder(status))


@app.post("/prediction")
def get_prediction(features: FeaturesData) -> int:
    """Get prediction from loaded model based on"""
    prediction = {"prediction": 1}
    values = []
    for _ in range(14):
        values.append(1)
    input_tensor = torch.tensor([values], dtype=torch.float32)
    with torch.no_grad():
        output = model(input_tensor)
    print(output)
    return JSONResponse(content=jsonable_encoder(prediction))

from pydantic import BaseModel


class FeaturesData(BaseModel):
    features: dict

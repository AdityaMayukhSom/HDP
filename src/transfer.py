from pydantic import BaseModel

from src.extractor import ExtractedFeatures


class SummarizeRequest(BaseModel):
    abstract: str


class SummarizeResponse(BaseModel):
    highlight: str


class GreetingResponse(BaseModel):
    message: str


class ExtractFeaturesRequest(BaseModel):
    abstract: str
    highlight: str


class ExtractFeaturesResponse(BaseModel):
    features: ExtractedFeatures

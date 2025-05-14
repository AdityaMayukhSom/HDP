from pydantic import BaseModel


class SummarizeRequest(BaseModel):
    abstract: str


class SummarizeResponse(BaseModel):
    highlight: str


class GreetingResponse(BaseModel):
    message: str

from fastapi import APIRouter
from fastapi.responses import JSONResponse

api_router = APIRouter(prefix="/api")


@api_router.get("/greetings/{name}", response_class=JSONResponse)
def hello_world(name: str):
    if name is None or len(name) == 0:
        name = "Stranger"

    return {
        "message": f"Hello {name} From FastAPI...",
    }

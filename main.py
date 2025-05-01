import sys

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from src.api import api_router
from src.web import web_router

app = FastAPI()
app.mount("/public", StaticFiles(directory="public"), name="public")
app.include_router(api_router)
app.include_router(web_router)


def main(argv: list[str]):
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=5000,
        log_level="info",
        reload=True,
        # ssl_keyfile="./certs-localhost/key.pem",
        # ssl_certfile="./certs-localhost/cert.pem",
    )


if __name__ == "__main__":
    main(sys.argv)

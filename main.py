import ssl
import sys

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

from src.api import api_router
from src.web import web_router

app = FastAPI()
app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=5)

app.mount(
    "/public",
    StaticFiles(directory="public", check_dir=False, follow_symlink=True),
    name="public",
)

app.include_router(api_router)
app.include_router(web_router)


def main(argv: list[str]):
    # ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    # ssl_context.load_cert_chain(
    #     certfile="/etc/ssl/certs/apache-selfsigned.crt",
    #     keyfile="/etc/ssl/private/apache-selfsigned.key",
    # )

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5000,
        log_level="debug",
        reload=True,
        use_colors=True,
        # ssl=ssl_context,
    )


if __name__ == "__main__":
    main(sys.argv)

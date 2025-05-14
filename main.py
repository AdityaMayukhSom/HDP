import ctypes
import gc
import sys

import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from utils.summarizer import generate_highlight, get_inference_model_and_tokenizer
from utils.transfer import GreetingResponse, SummarizeRequest, SummarizeResponse

IS_DEBUG = True
APP_PORT = 5000

app = FastAPI(root_path="/api", debug=IS_DEBUG, redirect_slashes=True)


@app.get("/greet/{name}", response_class=JSONResponse)
def greet_user(name: str):
    if name is None or len(name) == 0:
        name = "Stranger"
    return GreetingResponse(message=f"Hello {name.title()} From FastAPI...")


@app.post("/summarize", response_class=JSONResponse)
def summarize(request: SummarizeRequest):
    libc = ctypes.CDLL("libc.so.6")  # clearing cache

    libc.malloc_trim(0)
    gc.collect()
    torch.cuda.empty_cache()

    model, tokenizer = get_inference_model_and_tokenizer()
    highlight = generate_highlight(request.abstract, model, tokenizer)

    del model, tokenizer

    torch.cuda.empty_cache()
    gc.collect()
    libc.malloc_trim(0)

    return SummarizeResponse(highlight=highlight)


app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=5)

app.mount(
    "/",
    StaticFiles(directory="public", check_dir=False, follow_symlink=True, html=True),
    name="public",
)


def main(argv: list[str]):
    # ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    # ssl_context.load_cert_chain(
    #     certfile="/etc/ssl/certs/apache-selfsigned.crt",
    #     keyfile="/etc/ssl/private/apache-selfsigned.key",
    # )

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=APP_PORT,
        log_level="debug",
        reload=IS_DEBUG,
        use_colors=True,
        # ssl=ssl_context,
    )


if __name__ == "__main__":
    main(sys.argv)

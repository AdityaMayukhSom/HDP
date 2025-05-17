import uvicorn
from fastapi import FastAPI
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from utils.config import Config
from utils.extractor import LLMModel
from utils.lifespan import lifespan, mm, tm
from utils.memory import print_memory_stats
from utils.summarizer import generate_highlight
from utils.transfer import (
    ExtractFeaturesRequest,
    ExtractFeaturesResponse,
    GreetingResponse,
    SummarizeRequest,
    SummarizeResponse,
)

app = FastAPI(
    root_path="/api",
    debug=Config.IS_DEBUG,
    redirect_slashes=True,
    lifespan=lifespan,
)


@app.get("/greet/{name}", response_class=JSONResponse)
def greet_user(name: str) -> GreetingResponse:
    if name is None or len(name) == 0:
        name = "Stranger"
    return GreetingResponse(
        message=f"Hello {name.title()} From FastAPI...",
    )


@app.post("/extract/features", response_class=JSONResponse)
def calculate_evaluate_features(
    request: ExtractFeaturesRequest,
) -> ExtractFeaturesResponse:
    model: LLMModel = mm["extractor_llm"]

    features = model.extract_features(
        knowledge="",
        document=request.abstract,
        generated_text=request.highlight,
    )
    del model

    clf = mm["extractor_clf"]

    return ExtractFeaturesResponse(features=features)


@app.post("/summarize", response_class=JSONResponse)
def summarize(request: SummarizeRequest) -> SummarizeResponse:
    print_memory_stats()

    highlight = generate_highlight(
        request.abstract,
        mm["summarizer"],
        tm["summarizer"],
    )

    print_memory_stats()
    return SummarizeResponse(highlight=highlight)


if __name__ == "__main__":
    app.add_middleware(
        GZipMiddleware,
        minimum_size=1000,
        compresslevel=5,
    )

    app.mount(
        path="/",
        app=StaticFiles(
            directory="public",
            check_dir=False,
            follow_symlink=True,
            html=True,
        ),
        name="public",
    )
    # ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    # ssl_context.load_cert_chain(
    #     certfile="/etc/ssl/certs/apache-selfsigned.crt",
    #     keyfile="/etc/ssl/private/apache-selfsigned.key",
    # )

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=Config.APP_PORT,
        log_level=Config.LOG_LEVEL,
        reload=Config.IS_DEBUG,
        use_colors=True,
        # ssl=ssl_context,
    )

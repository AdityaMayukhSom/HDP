from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI

from utils.extractor import UnslothLLaMA, get_clf_and_std_scaler
from utils.memory import empty_all_memory, print_memory_stats
from utils.summarizer import get_summarizer_model_tokenizer

mm = dict[Literal["summarizer", "extractor_llm", "extractor_clf"], any]()
tm = dict[Literal["summarizer"], any]()
ss = dict[Literal["extractor_clf"], any]()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print_memory_stats()
    empty_all_memory()

    mm["summarizer"], tm["summarizer"] = get_summarizer_model_tokenizer()
    mm["extractor_llm"] = UnslothLLaMA()
    mm["extractor_clf"], ss["extractor_clf"] = get_clf_and_std_scaler()

    print_memory_stats()

    yield

    print_memory_stats()

    # extractor model will be moved to cpu when the model will be garbage collected
    # hence we only need to move the summarizer model to cpu before gc call
    mm["summarizer"].to("cpu")

    print_memory_stats()

    del mm["summarizer"]
    del mm["extractor_llm"]

    del mm["extractor_clf"]

    del tm["summarizer"]
    del ss["extractor_clf"]

    empty_all_memory()
    print_memory_stats()

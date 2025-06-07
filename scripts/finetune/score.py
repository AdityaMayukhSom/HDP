import json
import math
import os
import pathlib
import sys
import time

import evaluate
import pandas as pd
import sqlalchemy as sa
from loguru import logger
from tqdm import tqdm

sys.path.append(str(pathlib.Path(__file__).parent.parent))
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent))

from moverscore.moverscore_v2 import get_idf_dict, word_mover_score
from src.utils import get_mongo_client, get_mongo_database, get_postgresql_engine

os.environ["TZ"] = "Asia/Kolkata"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
time.tzset()


rouge = evaluate.load("rouge")  # []
bleu = evaluate.load("bleu")  # []
bleurt = evaluate.load("bleurt", "bleurt-large-512")  # []
google_bleu = evaluate.load("google_bleu")
# gleu = evaluate.load("gleu")  # []
meteor = evaluate.load("meteor")  # []
bert = evaluate.load("bertscore", lang="en")  # []
sari = evaluate.load("sari")


def compute_metrics(pred: list[str], real: list[str]):
    # https://www.freecodecamp.org/news/python-merge-dictionaries-merging-two-dicts-in-python/
    # https://www.datacamp.com/tutorial/python-dictionary-append

    scores: dict[str, str] = {}

    s_bleu = bleu.compute(
        predictions=pred,
        references=real,
    )

    for key, score in s_bleu.items():
        scores[f"bleu_{key}"] = score

    s_rouge = rouge.compute(
        predictions=pred,
        references=real,
    )

    for key, score in s_rouge.items():
        scores[f"rouge_{key}"] = score

    s_bert = bert.compute(
        predictions=pred,
        references=real,
        lang="en",
    )

    for key, score in s_bert.items():
        scores[f"bert_{key}"] = score

    s_bleurt = bleurt.compute(
        predictions=pred,
        references=real,
    )

    for key, score in s_bleurt.items():
        scores[f"bleurt_{key}"] = score

    s_google_bleu = google_bleu.compute(
        predictions=pred,
        references=real,
    )

    for key, score in s_google_bleu.items():
        scores[f"google_bleu_{key}"] = score

    s_meteor = meteor.compute(
        predictions=pred,
        references=real,
    )

    for key, score in s_meteor.items():
        scores[f"meteor_{key}"] = score

    idf_dict_real = get_idf_dict(real)
    idf_dict_pred = get_idf_dict(pred)

    s_mover = word_mover_score(
        real,
        pred,
        idf_dict_real,
        idf_dict_pred,
        stop_words=[],
        n_gram=1,
        remove_subwords=True,
    )

    scores["moverscore"] = s_mover

    return scores


_TOTAL_TODO_DATAPOINT_QUERY = """
SELECT COUNT("PII") AS "ToDo"
FROM
    (SELECT "PII"
     FROM "MixSub"
     WHERE "LlamaHighlight" IS NOT NULL
         AND "Split" = 'TEST'
     ORDER BY "PII"
     OFFSET :offset);
"""

_DATAPOINT_EXTRACTION_QUERY = """
SELECT ms."PII" AS pii,
       msv."CorrectHighlight" AS real,
       ms."LlamaHighlight" AS pred
FROM "MixSubView" msv
JOIN "MixSub" ms ON msv."PII" = ms."PII"
WHERE ms."LlamaHighlight" IS NOT NULL
    AND ms."Split" = 'TEST'
ORDER BY ms."PII" ASC
LIMIT :batch_size
OFFSET :offset;
"""


def main():
    logger.add(
        "./logs/scores/llama_{time:YYYY_MM_DD_hh_mm}.log",
        colorize=False,
        backtrace=True,
        diagnose=True,
        rotation="1 MB",
    )

    BATCH_SIZE = 4

    engine = get_postgresql_engine()

    with engine.connect() as conn, get_mongo_client() as client:
        mongodb = get_mongo_database(client)

        scores_collection = mongodb["LLAMA-SCORES"]
        done_batch_cnt = scores_collection.count_documents({})

        # NOTE: We're only extracting test split queries here for faster results
        res = (
            conn.execute(
                sa.text(_TOTAL_TODO_DATAPOINT_QUERY),
                {
                    "offset": done_batch_cnt * BATCH_SIZE,
                },
            )
            .mappings()
            .fetchone()
        )

        todo_cnt = int(res["ToDo"])
        iters = math.ceil(todo_cnt / BATCH_SIZE)
        logger.info(f"TODO: count {todo_cnt} datapoints, {iters} iterations")
        extraction_query = sa.text(_DATAPOINT_EXTRACTION_QUERY)

        for i in tqdm(range(done_batch_cnt, iters)):
            df = pd.read_sql_query(
                extraction_query,
                conn,
                params={
                    "batch_size": BATCH_SIZE,
                    "offset": i * BATCH_SIZE,
                },
            )

            s = compute_metrics(df["pred"], df["real"])

            doc = {
                "batch_index": i,
                "piis": df["pii"].to_list(),
                "scores": s,
            }

            logger.info(f"\n{json.dumps(doc, indent=2)}")
            scores_collection.insert_one(doc)


if __name__ == "__main__":
    main()

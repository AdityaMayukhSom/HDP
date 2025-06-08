import json
import os
import pathlib
import sys
import time
from datetime import datetime

import pandas as pd
import sqlalchemy as sa
import torch
from loguru import logger
from tqdm import tqdm

from unsloth import FastLanguageModel, is_bfloat16_supported  # isort:skip
from transformers import PreTrainedTokenizerFast  # isort:skip
from peft import PeftModelForCausalLM  # isort:skip

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from src.config import Config
from src.prompt import eval_summarizer_row_json_single_example
from src.utils import elapsed_time, get_postgresql_engine

os.environ["TZ"] = "Asia/Kolkata"
time.tzset()

logger.add(
    "./logs/summarize/qwen3_{time:YYYY_MM_DD_hh_mm}.log",
    colorize=False,
    backtrace=True,
    diagnose=True,
    rotation="1 MB",
)


def generate_eval_highlights(
    abstracts: list[str],
    model: PeftModelForCausalLM,
    tokenizer: PreTrainedTokenizerFast,
) -> list[str]:
    term_identifier = tokenizer.eos_token

    batch_data = [
        eval_summarizer_row_json_single_example(abstract) for abstract in abstracts
    ]

    inputs = tokenizer.apply_chat_template(
        batch_data,
        add_generation_prompt=True,
        tokenize=True,
        padding=True,
        truncation=True,
        return_tensors="pt",
        return_dict=True,
    ).to(Config.DEVICE)

    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    highlights = tokenizer.batch_decode(
        outputs[:, inputs.input_ids.shape[1] :],
        skip_special_tokens=True,
    )

    logger.info(f"\nRAW HIGHLIGHTS ::\n{highlights}")

    # the generaed output contains the prompt, init identifier, generated highlight and term identifier
    # the following code splits the output with init identifier, takes the second section which contains
    # the generated highlight followed by term identifier, now splits the second section based on term
    # identifier, takes the first section, which contains only the generated output. Then it strips the
    # generated content to get rid of any white spaces from the beginning and the end, and replaces
    # newline character with whitespace.

    highlights = [
        highlight.split(term_identifier)[0]
        .strip()
        .lstrip("<think>\n\n</think>\n\n")
        .replace("\n", " ")
        .strip()
        for highlight in highlights
    ]

    inputs.to("cpu")

    del batch_data
    del inputs

    return highlights


_TODO_DATAPOINT_QUERY = """\
SELECT COUNT(*) AS "ToDo"
FROM "MixSub"
WHERE "QwenHighlight" IS NULL
    AND "Split" = :split
"""

_DATAPOINT_EXTRACT_QUERY = """\
SELECT "MixSub"."PII" AS pii,
       "MixSubView"."ArticleAbstract" AS abstract
FROM "MixSubView"
JOIN "MixSub" ON "MixSubView"."PII" = "MixSub"."PII"
WHERE "MixSub"."QwenHighlight" IS NULL
    AND "MixSub"."Split" = :split
FETCH FIRST ROW ONLY
"""

_GENERATED_HIGHLIGHT_SAVE_QUERY = """\
UPDATE "MixSub" 
SET "QwenHighlight" = :highlight
WHERE "PII" = :pii
"""


def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="AdityaMayukhSom/Qwen3-1.7B-HyperMixSub-Full",
        max_seq_length=2048,
        load_in_4bit=True,
        dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16,
    )

    print(type(model))
    FastLanguageModel.for_inference(model)

    print(type(model))
    print(type(tokenizer))

    engine = get_postgresql_engine()

    SPLIT = "TEST"

    todo_query = sa.text(_TODO_DATAPOINT_QUERY)
    extract_query = sa.text(_DATAPOINT_EXTRACT_QUERY)
    insert_query = sa.text(_GENERATED_HIGHLIGHT_SAVE_QUERY)

    start_time = time.time()
    logger.info(f"summarizer model dataset generation started at {datetime.now()}")

    with engine.connect() as conn:
        todo_df = pd.read_sql_query(todo_query, conn, params={"split": SPLIT})
        todo_cnt = todo_df["ToDo"][0]

        logger.info(f"TODO: Generate Highlight For {todo_cnt} Data Points")

        for i in tqdm(range(todo_cnt)):
            df = pd.read_sql_query(
                extract_query,
                conn,
                params={
                    "split": SPLIT,
                },
            )

            if df.shape[0] == 0:
                break

            piis = df["pii"].to_list()
            abstracts = df["abstract"].to_list()

            highlights = generate_eval_highlights(abstracts, model, tokenizer)

            data = [
                {
                    "pii": pii,
                    "highlight": highlight,
                }
                for pii, highlight in zip(piis, highlights)
            ]

            logger.info(f"\n\n{json.dumps(data, indent=4)}")

            conn.execute(insert_query, data)
            conn.commit()

            logger.success(
                f"FINISHED :: {piis}\nSPLIT :: {SPLIT}\nTIME ELAPSED :: {elapsed_time(start_time)}"
            )

            break

    logger.success("summarizer model dataset generation finished")


if __name__ == "__main__":
    main()

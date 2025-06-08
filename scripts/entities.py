import json
import os
import pathlib
import sys
import threading
import time
from pathlib import Path
from textwrap import dedent

import sqlalchemy as sa
from google import genai
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent))

from src.utils import get_postgresql_engine, load_dotenv_in_config

os.environ["TZ"] = "Asia/Kolkata"
time.tzset()

logger.add(
    "./logs/entity/gemini_{time:YYYY_MM_DD_hh_mm}.log",
    colorize=False,
    backtrace=True,
    diagnose=True,
    rotation="5 MB",
)


def generate_gemini_prompt(instruction: str, document: str, summary: str):
    dat = {
        "Document": document,
        "Summary": summary,
    }

    prompt = f"{instruction}\n\n{json.dumps(dat, indent=2)}".strip()
    return prompt


def clean_json_markdown(md_text: str):
    json_pref = "```json"
    json_suff = "```"

    if md_text.startswith(json_pref):
        md_text = md_text.lstrip(json_pref)
    if md_text.endswith(json_suff):
        md_text = md_text.rstrip(json_suff)

    return md_text


def get_entities_from_gemini(
    instruction: str,
    document: str,
    summary: str,
    client: genai.Client,
):
    prompt = generate_gemini_prompt(instruction, document, summary)
    res = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    ent_json = (res.text if res.text is not None else "").strip()
    ent_json = clean_json_markdown(ent_json).strip()
    ent_json = json.dumps(json.loads(ent_json), indent=4)
    return ent_json


def generate_datapoint_entities(
    instruction: str,
    abstract: str,
    c_highlight: str,
    h_highlight: str,
    client: genai.Client,
) -> tuple[str, str]:
    # Correct Highlight Entities
    ch_ent = get_entities_from_gemini(instruction, abstract, c_highlight, client)

    # Hallucinated Highlight Entities
    hh_ent = get_entities_from_gemini(instruction, abstract, h_highlight, client)

    return ch_ent, hh_ent


_TODO_COUNT_QUERY = """\
SELECT COUNT(*) AS "TODO" 
FROM "MixSub" 
WHERE "HallucinatedHighlightEntities" IS NULL
"""

_DATAPOINT_EXTRACT_QUERY = """\
SELECT ms."PII" AS pii,
       msv."ArticleAbstract" AS abstract,
       msv."CorrectHighlight" AS c_highlight,
       msv."HallucinatedHighlight" AS h_highlight
FROM "MixSub" ms
JOIN "MixSubView" msv ON ms."PII" = msv."PII"
WHERE ms."HallucinatedHighlightEntities" IS NULL
FETCH FIRST ROW ONLY
FOR UPDATE SKIP LOCKED
"""

_UPDATE_ENTITIES_QUERY = """\
UPDATE "MixSub" 
SET "CorrectHighlightEntities" = :CorrectHighlightEntities,
    "HallucinatedHighlightEntities" = :HallucinatedHighlightEntities
WHERE "PII" = :PII
"""

_LOG_MSG_AND_ENT_FORMAT = """
PII :: {}

Correct Highlight:
{}

Correct Highlight Entities: 
{}

Hallucinated Highlight:
{}

Hallucinated Highlight Entities:
{}

STATUS :: SUCCESS
SLEEP :: {} SECONDS
"""


def extract_and_save(key: str, instr: str, engine: sa.Engine):
    logger.info(f"called extract and save for key {key}")
    client = genai.Client(api_key=key)

    extract_query = sa.text(_DATAPOINT_EXTRACT_QUERY)
    updt_query = sa.text(_UPDATE_ENTITIES_QUERY)

    updt_cnt = 0
    unit_delay = 120

    with engine.connect() as conn:
        while True:
            try:
                res = conn.execute(extract_query).mappings()
                d = res.fetchone()

                if d is None:
                    break
            except Exception as e:
                logger.error(f"ERROR DURING DB FETCH WITH API KEY :: {key}\n{str(e)}")
                conn.commit()
                break

            pii = d["pii"]
            abstract = d["abstract"]
            c_highlight = d["c_highlight"]
            h_highlight = d["h_highlight"]

            logger.info(f"STARTING EXTRACTION FOR PII :: {pii}")

            try:
                ch_ent, hh_ent = generate_datapoint_entities(
                    instr,
                    abstract,
                    c_highlight,
                    h_highlight,
                    client,
                )

                logger.success(
                    dedent(_LOG_MSG_AND_ENT_FORMAT).format(
                        pii,
                        c_highlight,
                        ch_ent,
                        h_highlight,
                        hh_ent,
                        unit_delay,
                    )
                )

                conn.execute(
                    updt_query,
                    {
                        "PII": pii,
                        "CorrectHighlightEntities": ch_ent,
                        "HallucinatedHighlightEntities": hh_ent,
                    },
                )

                updt_cnt += 1
            except Exception as e:
                logger.error(
                    f"ERROR: API KEY {key}, PII {pii}, SLEEP {unit_delay} SECONDS\n{str(e)}",
                )

                # put garbage value so that it won't be used later
                conn.execute(
                    updt_query,
                    {
                        "PII": pii,
                        "CorrectHighlightEntities": [],
                        "HallucinatedHighlightEntities": [],
                    },
                )
            finally:
                conn.commit()
                time.sleep(unit_delay)

    logger.success(
        f"FINISHED EXTRACTING ENTITIES WITH API KEY :: {key}, EXTRACTION COUNT :: {updt_cnt}"
    )


def main(argv: list[str]):
    engine = get_postgresql_engine()

    keys = pathlib.Path("keys.txt").read_text(encoding="utf-8").strip().split("\n")
    keys = set(filter(lambda x: x.strip() != "" and not x.startswith("#"), keys))

    with engine.connect() as conn:
        todo_query = sa.text(_TODO_COUNT_QUERY)
        res = conn.execute(todo_query).mappings().fetchone()
        todo_cnt = int(res["TODO"])

        logger.info(f"TODO :: EXTRACT ENTITIES FOR {todo_cnt} DATAPOINTS.")

    instr = pathlib.Path("./instructions/gemini-ner.txt").read_text(encoding="utf-8")

    # conf = load_dotenv_in_config()
    # extract_and_save(conf["GEMINI_API_KEY"], instr, engine)

    ts = [
        threading.Thread(target=extract_and_save, args=(key, instr, engine))
        for key in keys
    ]

    for t in ts:
        # Each thread starts with 1 second delay in between
        time.sleep(1)
        t.start()

    for t in ts:
        t.join()

    logger.success("FINISHED EXTRACTING ENTITIES")

    # with (
    #     open("./instructions/gemini-ner.txt", "r", encoding="utf-8") as instr_file,
    #     open("./data/example-abstract.txt", "r", encoding="utf-8") as abstract_file,
    #     open("./data/example-highlight.txt", "r", encoding="utf-8") as highlight_file,
    #     open("./data/gemini-prompt.txt", "w", encoding="utf-8") as prompt_file,
    # ):
    #     prompt = generate_gemini_prompt(
    #         instruction=instr_file.read().strip(),
    #         document=abstract_file.read().strip(),
    #         summary=highlight_file.read().strip(),
    #     )
    #     prompt_file.write(prompt)


if __name__ == "__main__":
    main(sys.argv)

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

from src.utils import get_postgresql_engine

os.environ["TZ"] = "Asia/Kolkata"
time.tzset()

logger.add(
    "./logs/gemini/outputfile_{time:YYYY_MM_DD_hh_mm}.log",
    colorize=False,
    backtrace=True,
    diagnose=True,
    rotation="1 MB",
)


def generate_gemini_prompt(
    instruction: str,
    abstract: str,
):
    dat = {
        "Document": abstract,
    }

    prompt = f"{instruction}\n\n{json.dumps(dat, indent=4)}".strip()
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
    client: genai.Client,
):
    prompt = generate_gemini_prompt(instruction, document)
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
    c_highlight: str,
    h_highlight: str,
    client: genai.Client,
) -> dict[str, str]:
    ch_ent = get_entities_from_gemini(instruction, c_highlight, client)
    hh_ent = get_entities_from_gemini(instruction, h_highlight, client)

    return {
        "CorrectHighlightEntities": ch_ent,
        "HallucinatedHighlightEntities": hh_ent,
    }


def extract_and_save(key: str, instr: str, engine: sa.Engine):
    logger.info(f"called extract and save for key {key}")
    client = genai.Client(api_key=key)

    extract_query = sa.text(
        """
        SELECT *
        FROM "MixSub"
        WHERE "HallucinatedHighlightEntities" IS NULL
        ORDER BY RANDOM() 
        FETCH FIRST ROW ONLY
        FOR UPDATE 
        SKIP LOCKED
        """
    )

    updt_query = sa.text(
        """
        UPDATE "MixSub" 
        SET "CorrectHighlightEntities" = :CorrectHighlightEntities,
            "HallucinatedHighlightEntities" = :HallucinatedHighlightEntities
        WHERE "PII" = :PII
        """
    )

    updt_cnt = 0
    unit_delay = 180

    with engine.connect() as conn:
        while True:
            try:
                res = conn.execute(extract_query).mappings()
                d = res.fetchone()

                if d is None:
                    break
            except Exception as e:
                logger.error(f"\nError with key {key} during fetch\n{str(e)}")
                conn.commit()
                break

            pii = d["PII"]
            abstract = (
                d["BetterAbstract"]
                if d["BetterAbstract"] is not None
                else d["OriginalAbstract"]
            )
            c_highlight = (
                d["BetterHighlight"]
                if d["BetterHighlight"] is not None
                else d["OriginalHighlight"]
            )
            h_highlight = d["HallucinatedHighlight"]
            logger.info(f"starting extraction for {pii}")

            dat = {
                "PII": pii,
                "CorrectHighlightEntities": [],
                "HallucinatedHighlightEntities": [],
            }

            try:
                res = generate_datapoint_entities(
                    instr,
                    c_highlight,
                    h_highlight,
                    client,
                )
            except Exception as e:
                logger.error(f"\nError with key {key}\n{str(e)}")
                logger.warning(f"PII: {pii}, STATUS: FAILED, SLEEP {unit_delay} SEC...")
                time.sleep(unit_delay)

                # put garbage value so that it won't be used later
                conn.execute(updt_query, dat)
                conn.commit()
                continue

            dat = {
                "PII": pii,
                "CorrectHighlightEntities": res["CorrectHighlightEntities"],
                "HallucinatedHighlightEntities": res["HallucinatedHighlightEntities"],
            }

            msg = """
            PII: {}
            
            Correct Highlight:
            {}

            Correct Highlight Entities: 
            {}
            
            Hallucinated Highlight:
            {}

            Hallucinated Highlight Entities:
            {}
            
            STATUS: SUCCESS
            SLEEP {} SEC...
            """

            log_msg = dedent(msg).format(
                pii,
                c_highlight,
                res["CorrectHighlightEntities"],
                h_highlight,
                res["HallucinatedHighlightEntities"],
                unit_delay,
            )
            logger.success(log_msg)

            try:
                # conn.execute(updt_query, dat)
                updt_cnt += 1
            except Exception as e:
                logger.error(f"\nError with key {key}\n{str(e)}")
                logger.warning(f"PII: {pii}, STATUS: FAILED, SLEEP {unit_delay} SEC...")
            finally:
                conn.commit()
                time.sleep(unit_delay)

            break

    logger.success(
        f"finished generating hallucinated highlights for key {key} with update count {updt_cnt}"
    )


def main(argv: list[str]):
    engine = get_postgresql_engine()
    keys = pathlib.Path("keys.txt").read_text(encoding="utf-8").strip().split("\n")
    keys = set(filter(lambda x: x.strip() != "", keys))

    instr = pathlib.Path("./instructions/gemini-ner.txt").read_text(encoding="utf-8")

    with engine.connect() as conn:
        todo_query = sa.text(
            """
            SELECT COUNT(*) AS "TODO" 
            FROM "MixSub" 
            WHERE "HallucinatedHighlightEntities" IS NULL
            """
        )
        res = conn.execute(todo_query).mappings().fetchone()
        todo_cnt = int(res["TODO"])

        logger.info(f"TODO: generate hallucinated highlight for {todo_cnt} datapoints")

    extract_and_save("AIzaSyB6nVwHOQiRvuBIp97azwisiqefxykGZGg", instr, engine)

    # ts = [
    #     threading.Thread(target=extract_and_save, args=(key, instr, engine))
    #     for key in keys
    # ]

    # for t in ts:
    #     t.start()

    # for t in ts:
    #     t.join()

    logger.success("finished hallucinated highlight generation")

    # with (
    #     open("./instructions/gemini.txt", "r", encoding="utf-8") as instruction_file,
    #     open("./data/example-abstract.txt", "r", encoding="utf-8") as abstract_file,
    #     open("./data/example-highlight.txt", "r", encoding="utf-8") as highlight_file,
    #     open("./data/gemini-prompt.txt", "w", encoding="utf-8") as prompt_file,
    # ):
    #     prompt = generate_hallucinated_highlight(
    #         instruction=instruction_file.read(),
    #         abstract=abstract_file.read(),
    #         correct_highlight=highlight_file.read(),
    #     )
    #     prompt_file.write(prompt)


if __name__ == "__main__":
    main(sys.argv)

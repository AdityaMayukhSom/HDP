import json
import os
import pathlib
import sys
import threading
import time

import sqlalchemy as sa
from dotenv import dotenv_values
from google import genai
from loguru import logger

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
    correct_highlight: str,
):
    dat = {
        "Document": abstract,
        "CorrectHighlight": correct_highlight,
    }

    prompt = f"{instruction}\n\n{json.dumps(dat, indent=4)}".strip()
    return prompt


def generate_hallucinated_highlight(
    instruction: str,
    abstract: str,
    correct_highlight: str,
    client: genai.Client,
) -> dict[str, str]:
    prompt = generate_gemini_prompt(instruction, abstract, correct_highlight)

    res = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )

    return {
        "HallucinatedHighlight": " ".join(res.text.split("\n")),
    }


def extract_and_save(key: str, instr: str, engine: sa.Engine):
    logger.info(f"called extract and save for key {key}")
    client = genai.Client(api_key=key)

    extract_query = sa.text(
        """
        SELECT * FROM "MixSubView" 
        WHERE "HallucinatedHighlight" IS NULL
        FETCH FIRST ROW ONLY
        FOR UPDATE SKIP LOCKED
        """
    )

    updt_query = sa.text(
        """
        UPDATE "MixSub" 
        SET "HallucinatedHighlight" = :hallucinated_highlight 
        WHERE "PII" = :pii
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

                pii = d["PII"]
                abstract = d["ArticleAbstract"]
                correct = d["CorrectHighlight"]
                hallucinated = ""
            except Exception as e:
                logger.error(f"\nError with key {key} during fetch\n{str(e)}")
                break

            logger.info(f"starting extraction for {pii}")

            try:
                res = generate_hallucinated_highlight(
                    instr,
                    abstract,
                    correct,
                    # modulus is unnecessary, as we only extract
                    client,
                )

                hallucinated = res["HallucinatedHighlight"]

                conn.execute(
                    updt_query,
                    {
                        "pii": pii,
                        "hallucinated_highlight": hallucinated,
                    },
                )

                updt_cnt += 1
                logger.success(
                    f"\nPII: {pii}\n\nCorrect Summary:\n{correct}\n\nHallucinated Summary:\n{hallucinated}\n\nStatus: SUCCESS\nSleeping for {unit_delay} seconds..."
                )
            except Exception as e:
                logger.error(f"\nError with key {key}\n{str(e)}")
                logger.warning(
                    f"PII: {pii},Status: FAILED, Sleep {unit_delay} seconds."
                )
            finally:
                conn.commit()

            time.sleep(unit_delay)

    logger.success(
        f"finished generating hallucinated highlights for key {key} with update count {updt_cnt}"
    )


def main(argv: list[str]):
    config = dotenv_values(dotenv_path=".env", verbose=True, encoding="utf-8")
    keys = list(
        filter(
            lambda x: x.strip() != "",
            pathlib.Path("keys.txt").read_text(encoding="utf-8").strip().split("\n"),
        )
    )

    keys = set(keys)

    conn_url = sa.URL.create(
        drivername="postgresql+psycopg2",
        username=config["PG_USERNAME"],
        password=config["PG_PASSWORD"],
        host=config["PG_HOST"],
        database=config["PG_DATABASE"],
        port=int(config["PG_PORT"]),
    )

    instr = pathlib.Path("./scripts/instructions/gemini.txt").read_text(
        encoding="utf-8"
    )
    engine = sa.create_engine(conn_url, pool_size=128, max_overflow=256)

    with engine.connect() as conn:
        todo_query = sa.text(
            'SELECT COUNT(*) AS "TODO" FROM "MixSubView" WHERE "HallucinatedHighlight" IS NULL'
        )
        res = conn.execute(todo_query).mappings().fetchone()
        todo_cnt = int(res["TODO"])

        logger.info(f"TODO: generate hallucinated highlight for {todo_cnt} datapoints")

    ts = [
        threading.Thread(target=extract_and_save, args=(key, instr, engine))
        for key in keys
    ]

    for t in ts:
        t.start()

    for t in ts:
        t.join()

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

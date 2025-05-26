import json
import os
import pathlib
import sys
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


def generate_hallucinated_highlight(
    instruction: str,
    abstract: str,
    correct_highlight: str,
    client: genai.Client,
) -> dict[str, str]:
    dat = {
        "Document": abstract,
        "CorrectHighlight": correct_highlight,
    }

    mes = f"{instruction}\n\n{json.dumps(dat, indent=4)}".strip()

    res = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=mes,
    )

    return {
        "Message": mes,
        "HallucinatedHighlight": " ".join(res.text.split("\n")),
    }


def main(argv: list[str]):
    config = dotenv_values(dotenv_path=".env", verbose=True, encoding="utf-8")
    keys = list(
        filter(
            lambda x: x.strip() != "",
            pathlib.Path("keys.txt").read_text(encoding="utf-8").strip().split("\n"),
        )
    )

    conn_url = sa.URL.create(
        drivername="postgresql+psycopg2",
        username=config["PG_USERNAME"],
        password=config["PG_PASSWORD"],
        host=config["PG_HOST"],
        database=config["PG_DATABASE"],
        port=int(config["PG_PORT"]),
    )

    instr = pathlib.Path("./scripts/instruction/gemini.txt").read_text(encoding="utf-8")
    engine = sa.create_engine(conn_url)

    with engine.connect() as conn:
        todo_query = sa.text(
            'SELECT COUNT(*) AS "TODO" FROM "MixSubView" WHERE "HallucinatedHighlight" IS NULL'
        )
        res = conn.execute(todo_query).mappings().fetchone()
        todo_cnt = int(res["TODO"])

        logger.info(f"TODO: generate hallucinated highlight for {todo_cnt} datapoints")

        extract_query = sa.text(
            """
            SELECT * FROM "MixSubView" 
            WHERE "HallucinatedHighlight" IS NULL 
            LIMIT :limit
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
        clients = [genai.Client(api_key=key.strip()) for key in keys]

        while updt_cnt < todo_cnt:
            res = conn.execute(extract_query, {"limit": len(clients)}).mappings()
            data = res.fetchall()

            piis = [d["PII"] for d in data]
            logger.info(f"starting batch with count {len(data)} and values {piis}")

            for i, d in enumerate(data):
                pii = d["PII"]
                abstract = d["ArticleAbstract"]
                correct = d["CorrectHighlight"]
                hallucinated = ""

                try:
                    res = generate_hallucinated_highlight(
                        instr,
                        abstract,
                        correct,
                        clients[i % len(clients)],
                    )

                    hallucinated = res["HallucinatedHighlight"]
                except Exception as e:
                    logger.error(str(e))

                logger.info(
                    f"\nPII: {pii} ------------------\nCorrect Summary:\n{correct}\nHallucinated Summary:\n{hallucinated}\n"
                )

                conn.execute(
                    updt_query,
                    {
                        "pii": pii,
                        "hallucinated_highlight": hallucinated,
                    },
                )

                conn.commit()

                updt_cnt += 1

                time.sleep(1)

    logger.success("finished generating hallucinated highlights")

    # with (
    #     open("./scripts/instruction/gemini.txt", "r", encoding="utf-8") as ins_file,
    #     open("./scripts/example/abstract.txt", "r", encoding="utf-8") as abs_file,
    #     open("./scripts/example/highlight.txt", "r", encoding="utf-8") as hlt_file,
    # ):
    #     hal = generate_hallucinated_highlight(
    #         ins_file.read(),
    #         abs_file.read(),
    #         hlt_file.read(),
    #         client,
    #     )

    # with open("./scripts/example/message.txt", "w", encoding="utf-8") as mes_file:
    #     mes_file.write(hal["Message"])

    # with open("./scripts/instruction/oneturn.json", "r", encoding="utf-8") as f:
    #     instruction = json.load(f)


if __name__ == "__main__":
    main(sys.argv)

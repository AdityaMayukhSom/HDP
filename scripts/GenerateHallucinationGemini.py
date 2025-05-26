import json
import pathlib

import sqlalchemy as sa
from dotenv import dotenv_values
from google import genai
from loguru import logger


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
        "HallucinatedHighlight": res.text,
    }


if __name__ == "__main__":
    config = dotenv_values(dotenv_path=".env", verbose=True, encoding="utf-8")
    keys = config["GEMINI_API_KEY"].split(",")
    clients = [genai.Client(key.strip()) for key in keys]

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
            'SELECT COUNT(*) AS "TODO" FROM "MixSubView" WHERE "HallucinatedHighlight" IS NOT NULL'
        )
        res = conn.execute(todo_query).fetchone()
        todo_cnt = int(res["TODO"])

        logger.info(f"todo: generate hallucinated highlight for {todo_cnt} datapoints")

        extract_query = sa.text(
            'SELECT * FROM "MixSubView" WHERE "HallucinatedHighlight" IS NOT NULL LIMIT :limit'
        )

        updt_query = sa.text(
            """
            UPDATE "MixSub" 
            SET "HallucinatedHighlight" = :hallucinated_highlight 
            WHERE "PII" = :pii
            """
        )

        updt_cnt = 0

        while updt_cnt < todo_cnt:
            res = conn.execute(extract_query, {"limit": len(clients)})
            data = res.fetchall()

            for i, d in enumerate(data):
                hal = generate_hallucinated_highlight(
                    instr,
                    d["ArticleAbstract"],
                    d["CorrectHighlight"],
                    clients[i % len(clients)],
                )

                conn.execute(
                    updt_query,
                    {
                        "hallucinated_highlight": hal["HallucinatedHighlight"],
                        "pii": d["PII"],
                    },
                )

                updt_cnt += 1

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

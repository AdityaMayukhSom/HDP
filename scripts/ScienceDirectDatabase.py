import os
import time
from typing import Literal

import pandas as pd
import sqlalchemy
from datasets import DatasetDict, load_dataset
from dotenv import dotenv_values

os.environ["TZ"] = "Asia/Kolkata"
time.tzset()


def load_mix_sub_into_database(engine: sqlalchemy.Engine):
    # https://huggingface.co/docs/datasets/en/loading#hugging-face-hub
    ds: DatasetDict[Literal["train", "validation", "test"]] = load_dataset(
        "TRnlp/MixSub"
    )

    # Changing all the column names to have uniform singular forms
    # All column names are now in singular form
    ds = ds.rename_column("Highlights", "Highlight")

    # execute the following lines to train the model on the entire dataset.
    trn_ds, val_ds, tst_ds = ds["train"], ds["validation"], ds["test"]

    trn_ds = trn_ds.map(lambda _: {"Split": "TRAIN"})
    val_ds = val_ds.map(lambda _: {"Split": "VALIDATION"})
    tst_ds = tst_ds.map(lambda _: {"Split": "TEST"})

    with engine.connect() as conn:
        trn_ds.to_sql("MixSub", conn, if_exists="append")
        val_ds.to_sql("MixSub", conn, if_exists="append")
        tst_ds.to_sql("MixSub", conn, if_exists="append")


def extract_broken_abstracts(engine: sqlalchemy.Engine):
    query = """
    SELECT
      "PII",
      "Split",
      'www.sciencedirect.com/science/article/abs/pii/' || "PII" AS "ScienceDirectLink"
    FROM
      "MixSub"
    WHERE
      "OriginalAbstract" NOT LIKE '%%.';
    """

    with (
        engine.connect() as conn,
        pd.ExcelWriter(
            "./scripts/BrokenAbstracts.xlsx",
            engine="openpyxl",
            mode="w",
        ) as writer,
    ):
        df = pd.read_sql(query, conn)
        df.to_excel(writer, sheet_name="MixSub", index=False)


def store_piis(engine: sqlalchemy.Engine):
    stmt = sqlalchemy.text(
        'INSERT INTO "MixSub" ("PII", "Split") VALUES (:pii, :split) ON CONFLICT DO NOTHING'
    )

    with (
        engine.connect() as conn,
        open("./data/sciencedirect.txt", "r", encoding="utf-8") as f,
    ):
        prefix = "https://www.sciencedirect.com/science/article/pii/"

        pii_params = []

        for line in f.readlines():
            line = line.strip()

            if line == "":
                continue

            # to handle bullet names
            pii = line.split()[-1].strip()

            if pii.startswith(prefix):
                pii = pii.strip(prefix)

            param = {
                "pii": pii.strip(),
                "split": "TRAIN",
            }

            pii_params.append(param)

        conn.execute(stmt, pii_params)
        conn.commit()


if __name__ == "__main__":
    config = dotenv_values(".env")

    conn_url = sqlalchemy.URL.create(
        drivername="postgresql+psycopg2",
        username=config["PG_USERNAME"],
        password=config["PG_PASSWORD"],
        host=config["PG_HOST"],
        database=config["PG_DATABASE"],
        port=int(config["PG_PORT"]),
    )

    engine = sqlalchemy.create_engine(conn_url)

    # init_mixsub_database(engine)
    # extract_broken_abstracts(engine)
    store_piis(engine)

    engine.dispose()

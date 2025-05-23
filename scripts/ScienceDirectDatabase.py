from typing import Literal

import pandas as pd
import sqlalchemy
from datasets import DatasetDict, load_dataset
from dotenv import dotenv_values


def get_mixsub():
    DatasetDictKeys = Literal["train", "validation", "test"]

    # https://huggingface.co/docs/datasets/en/loading#hugging-face-hub
    ds: DatasetDict[DatasetDictKeys] = load_dataset("TRnlp/MixSub")

    # Changing all the column names to have uniform singular forms
    # All column names are now in singular form
    ds = ds.rename_column("Highlights", "Highlight")

    # execute the following lines to train the model on the entire dataset.
    trn_ds, val_ds, tst_ds = ds["train"], ds["validation"], ds["test"]

    return trn_ds, val_ds, tst_ds


def populate_database(engine: sqlalchemy.Engine):
    trn_ds, val_ds, tst_ds = get_mixsub()

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
      "Filename",
      "Abstract",
      "Highlight",
      "Split",
      'https://www.sciencedirect.com/science/article/abs/pii/' || "Filename" AS "ScienceDirectLink"
    FROM
      "MixSub"
    WHERE
      "Abstract" NOT LIKE '%%.';
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


def store_article_ids(engine: sqlalchemy.Engine):
    def process_id(filename: str):
        return {
            "filename": filename.strip(),
            "highlight": "ADDED_MANUALLY",
            "abstract": "ADDED_MANUALLY_2",
            "split": "TRAIN",
        }

    stmt = sqlalchemy.text("""
    INSERT INTO "MixSub" ("Filename", "Highlight", "Abstract", "Split") 
    VALUES (:filename, :highlight, :abstract, :split)
    ON CONFLICT DO NOTHING
    """)

    with (
        engine.connect() as conn,
        open("./scripts/names.txt", "r", encoding="utf-8") as f,
    ):
        prefix = "https://www.sciencedirect.com/science/article/pii/"
        name_list = list(
            map(
                lambda x: (
                    x.strip().strip(prefix) if x.startswith(prefix) else x.strip()
                ),
                f.readlines(),
            )
        )
        name_params = list(map(process_id, name_list))
        conn.execute(stmt, name_params)
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

    # populate_database(engine)
    # extract_broken_abstracts(engine)
    store_article_ids(engine)

    engine.dispose()

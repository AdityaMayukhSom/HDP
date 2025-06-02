import os
import sys
import time
from pathlib import Path
from typing import Literal

import pandas as pd
import sqlalchemy as sa
from datasets import DatasetDict, load_dataset
from loguru import logger

sys.path.append(str(Path(__file__).parent.parent))
from src.utils import get_postgresql_engine

os.environ["TZ"] = "Asia/Kolkata"
time.tzset()


def load_mix_sub_into_database(engine: sa.Engine):
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


def extract_broken_abstracts(engine: sa.Engine):
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


def store_piis(engine: sa.Engine):
    stmt = sa.text(
        'INSERT INTO "MixSub" ("PII", "Split") VALUES (:pii, :split) ON CONFLICT DO NOTHING'
    )

    with (
        engine.connect() as conn,
        open("./data/sciencedirect.txt", "r", encoding="utf-8") as f,
    ):
        prefix_1 = "https://www.sciencedirect.com/science/article/pii/"
        prefix_2 = "https://www.sciencedirect.com/science/article/abs/pii/"

        pii_params = []

        for line in f.readlines():
            line = line.strip()

            if line == "":
                continue

            # to handle bullet names
            pii = line.split()[-1].strip()

            if pii.startswith(prefix_1):
                pii = pii.strip(prefix_1)
            elif pii.startswith(prefix_2):
                pii = pii.strip(prefix_2)

            pii = pii.strip()

            if len(pii) != 17:
                logger.error(f"PII length not 17 : {pii}")
            else:
                param = {
                    "pii": pii,
                    "split": "TRAIN",
                }

                pii_params.append(param)

        conn.execute(stmt, pii_params)
        conn.commit()


def main():
    engine = get_postgresql_engine()

    # load_mix_sub_into_database(engine)
    # extract_broken_abstracts(engine)
    store_piis(engine)

    engine.dispose()


if __name__ == "__main__":
    main()

import pandas as pd
import sqlalchemy as sa
from datasets import Dataset, DatasetDict


def extract_hyper_mix_sub_from_db(engine: sa.Engine) -> DatasetDict:
    extract_query = sa.text(
        """
        SELECT "PII", "ArticleAbstract", "CorrectHighlight"
        FROM "MixSubView"
        WHERE "Split" = :split;
        """
    )

    with engine.connect() as conn:
        trn_df = pd.read_sql(extract_query, conn, params={"split": "TRAIN"})
        val_df = pd.read_sql(extract_query, conn, params={"split": "VALIDATION"})
        tst_df = pd.read_sql(extract_query, conn, params={"split": "TEST"})

    # trn_df = pd.concat([trn_df, val_df, tst_df], ignore_index=True)

    trn_ds = Dataset.from_pandas(trn_df).formatted_as("torch")
    val_ds = Dataset.from_pandas(val_df).formatted_as("torch")
    tst_ds = Dataset.from_pandas(tst_df).formatted_as("torch")

    dd = DatasetDict(
        {
            "train": trn_ds,
            "validation": val_ds,
            "test": tst_ds,
        }
    )

    return dd

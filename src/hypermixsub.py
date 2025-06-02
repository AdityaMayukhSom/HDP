import pandas as pd
import sqlalchemy as sa
from datasets import Dataset, DatasetDict
from loguru import logger

from src.utils import get_postgresql_engine, load_dotenv_in_config


def extract_hms_from_db(
    engine: sa.Engine,
    query='SELECT "PII", "ArticleAbstract", "CorrectHighlight", "HallucinatedHighlight" FROM "MixSubView" WHERE "Split" = :split',
) -> DatasetDict:
    extract_query = sa.text(query)

    with engine.connect() as conn:
        trn_df = pd.read_sql(extract_query, conn, params={"split": "TRAIN"})
        val_df = pd.read_sql(extract_query, conn, params={"split": "VALIDATION"})
        tst_df = pd.read_sql(extract_query, conn, params={"split": "TEST"})

    logger.info(f"extracted {len(trn_df)} train rows")
    logger.info(f"extracted {len(val_df)} validation rows")
    logger.info(f"extracted {len(tst_df)} test rows")
    logger.info(f"extracted {len(trn_df) + len(val_df) + len(tst_df)} total rows")

    # trn_df = pd.concat([trn_df, val_df, tst_df], ignore_index=True)

    trn_ds = Dataset.from_pandas(trn_df)
    val_ds = Dataset.from_pandas(val_df)
    tst_ds = Dataset.from_pandas(tst_df)

    dd = DatasetDict(
        {
            "TRAIN": trn_ds,
            "VALIDATION": val_ds,
            "TEST": tst_ds,
        }
    )

    return dd


def insert_generated_highlight_into_db(
    engine: sa.Engine,
    piis: list[str],
    highlights: list[str],
):
    insert_query = sa.text(
        """
        UPDATE "MixSub" 
        SET "ModelGeneratedHighlight" = :highlight
        WHERE "PII" = :pii
        """
    )

    data = [
        {
            "pii": pii,
            "highlight": highlight,
        }
        for pii, highlight in zip(piis, highlights)
    ]

    # logger.info(f"inserting {len(data)} rows\n\n{data}")

    with engine.connect() as conn:
        conn.execute(insert_query, data)
        conn.commit()


def publish_hms_to_hf():
    config = load_dotenv_in_config()
    engine = get_postgresql_engine()
    ds = extract_hms_from_db(engine=engine)

    ds.push_to_hub(
        repo_id=config["HF_HMS_REPO_ID"],
        token=config["HF_TOKEN"],
        private=False,
    )


if __name__ == "__main__":
    publish_hms_to_hf()

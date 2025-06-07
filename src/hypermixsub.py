import pathlib
import sys

import pandas as pd
import sqlalchemy as sa
from datasets import Dataset, DatasetDict
from loguru import logger

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from src.utils import get_postgresql_engine, load_dotenv_in_config


def extract_hms_from_db(
    engine: sa.Engine,
    *,
    query: str | None = None,
    load_full_in_train=False,
) -> DatasetDict:
    if query is None:
        query = """\
        SELECT "PII", "ArticleAbstract", "CorrectHighlight", "HallucinatedHighlight" 
        FROM "MixSubView" 
        WHERE "Split" = :split
        """

    extract_query = sa.text(query)

    with engine.connect() as conn:
        trn_df = pd.read_sql(extract_query, conn, params={"split": "TRAIN"})
        val_df = pd.read_sql(extract_query, conn, params={"split": "VALIDATION"})
        tst_df = pd.read_sql(extract_query, conn, params={"split": "TEST"})

    logger.info(f"extracted {len(trn_df)} train rows")
    logger.info(f"extracted {len(val_df)} validation rows")
    logger.info(f"extracted {len(tst_df)} test rows")
    logger.info(f"extracted {len(trn_df) + len(val_df) + len(tst_df)} total rows")

    if load_full_in_train:
        trn_df = pd.concat([trn_df, val_df, tst_df], ignore_index=True)

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


def publish_hms_to_hf():
    config = load_dotenv_in_config()
    engine = get_postgresql_engine()
    ds = extract_hms_from_db(engine)

    ds.push_to_hub(
        repo_id=config["HF_HMS_REPO_ID"],
        token=config["HF_TOKEN"],
        private=False,
        commit_message="added hallucinated highlight for new datapoints",
    )


if __name__ == "__main__":
    publish_hms_to_hf()

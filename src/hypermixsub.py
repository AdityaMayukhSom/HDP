import pathlib
import sys

import pandas as pd
import sqlalchemy as sa
from datasets import Dataset, DatasetDict
from loguru import logger

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from src.utils import get_postgresql_engine, load_dotenv_in_config

_HMS_DB_EXTRACTION_QUERY = """\
SELECT "PII",
       "ArticleAbstract",
       "CorrectHighlight",
       "HallucinatedHighlight"
FROM "MixSubView"
WHERE "Split" = :split;
"""


def extract_hms_from_db(
    engine: sa.Engine,
    *,
    query: str | None = None,
    load_full_in_train=False,
) -> DatasetDict:
    if query is None:
        query = _HMS_DB_EXTRACTION_QUERY

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


def publish_hms_to_hf(commit_message: str):
    config = load_dotenv_in_config()
    engine = get_postgresql_engine()
    ds = extract_hms_from_db(engine)

    ds.push_to_hub(
        repo_id=config["HF_MSV2_REPO_ID"],
        token=config["HF_TOKEN"],
        private=False,
        commit_message=commit_message,
    )


_EXTRACT_ENTITES_FOR_NER_JSON_QUERY = """\
SELECT "PII",
       "ArticleAbstract",
       "Split",
       "CorrectHighlight",
       "CorrectHighlightEntities",
       "HallucinatedHighlight",
       "HallucinatedHighlightEntities"
FROM "MixSubView"
WHERE "HallucinatedHighlightEntities" IS NOT NULL
    AND "HallucinatedHighlightEntities" != '[]'
ORDER BY "PII" ASC
LIMIT :limit
OFFSET :offset
"""


def extract_entities_to_json(limit: int, offset: int, json_path: str):
    engine = get_postgresql_engine()

    q = sa.text(_EXTRACT_ENTITES_FOR_NER_JSON_QUERY)

    with engine.connect() as conn:
        df = pd.read_sql_query(
            q,
            conn,
            params={
                "limit": limit,
                "offset": offset,
            },
        )

        df.to_json(json_path, orient="records", indent=4)


def main():
    # extract_entities_to_json(400, 000, "./data/ner/debdut-hira-ner-01.json")
    # extract_entities_to_json(400, 400, "./data/ner/debdut-hira-ner-02.json")
    # extract_entities_to_json(400, 800, "./data/ner/debdut-hira-ner-03.json")
    # extract_entities_to_json(400, 1200, "./data/ner/priyanka-dey-ner-01.json")
    # extract_entities_to_json(400, 1600, "./data/ner/priyanka-dey-ner-02.json")
    # extract_entities_to_json(400, 2000, "./data/ner/priyanka-dey-ner-03.json")
    # extract_entities_to_json(400, 2400, "./data/ner/priyanka-dey-ner-04.json")
    # extract_entities_to_json(400, 2800, "./data/ner/priyanka-dey-ner-05.json")
    # extract_entities_to_json(400, 3200, "./data/ner/priyanka-dey-ner-06.json")
    # publish_hms_to_hf("trimmed left tail of length distribution")
    return


if __name__ == "__main__":
    main()

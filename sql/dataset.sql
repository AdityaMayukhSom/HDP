-- SELECT w.W AS "WITH_FILTER",
--        v.V AS "WITHOUT_FILTER",
--        CAST(w.W AS DECIMAL(7, 2)) * 100 / v.V AS "PERCENTAGE",
--        100 - PERCENTAGE AS "LOST"
-- FROM
--     (SELECT COUNT(*) AS W
--      FROM train
--      WHERE LENGTH (Highlights) <= 1024 ) AS w,
--
--     (SELECT COUNT(*) AS V
--      FROM train) AS v;
--
-- --

SELECT COUNT(*)
FROM "MixSub";


DELETE
FROM "MixSub"
WHERE "PII" IN ('S2212054825000177',
                'S2212054825000219');


SELECT COUNT(*)
FROM "MixSubView"
WHERE LENGTH("CorrectHighlight") > 1000
    OR LENGTH("ArticleAbstract") > 2800;


SELECT COUNT(*)
FROM "MixSubView"
WHERE LENGTH("CorrectHighlight") >= LENGTH("ArticleAbstract");


SELECT *
FROM "MixSubView"
ORDER BY RANDOM()
LIMIT 3;


DELETE
FROM "MixSub"
WHERE "PII" IN ('S2666498425000146',
                'S2666498425000365',
                'S2666498425000237');


INSERT INTO "MixSub" ("PII",
                      "Split")
VALUES ('S2666498425000146', 'TRAIN'),
       ('S2666498425000365', 'TRAIN'),
       ('S2666498425000237', 'TRAIN');


SELECT *
FROM "MixSub"
WHERE "PII" IN ('S2666498425000146',
                'S2666498425000365',
                'S2666498425000237');


SELECT *
FROM "MixSubView"
WHERE "CorrectHighlight" IS NULL
    OR "CorrectHighlight" = '';


SELECT *
FROM "MixSub"
WHERE "BetterHighlight" IS NULL
    AND "OriginalHighlight" IS NULL;


SELECT COUNT(*)
FROM "MixSub"
WHERE "Split" = 'VALIDATION';


SELECT A.P + B.P + C.P AS TOTAL
FROM
    (SELECT COUNT(*) AS P
     FROM train) AS A,

    (SELECT COUNT(*) AS P
     FROM test) AS B,

    (SELECT COUNT(*) AS P
     FROM validation) AS C;

--

SELECT *
FROM "MixSub"
LIMIT 10;

--

SELECT "Filename",
       "Abstract",
       "Highlight",
       "Split",
       'https://www.sciencedirect.com/science/article/abs/pii/' || "Filename" AS "ScienceDirectLink"
FROM "MixSub"
WHERE "Abstract" NOT LIKE '%.';

--

SELECT STRING_AGG("SUB"."SDLINK", ' ')
FROM
    (SELECT 'https://www.sciencedirect.com/science/article/abs/pii/' || "Filename" AS "SDLINK"
     FROM "MixSub"
     WHERE "Abstract" NOT LIKE '%.'
         AND "Split" = 'TRAIN'
     LIMIT 100) AS "SUB";


SELECT "Filename",
       "BetterAbstract",
       "BetterHighlight"
FROM "MixSub"
WHERE LENGTH("BetterAbstract") > 0
    AND LENGTH("BetterHighlight") > 0
LIMIT 10;


ALTER TABLE "MixSub" ADD COLUMN "BetterHighlight" TEXT;


ALTER TABLE "MixSub" ADD COLUMN "BetterAbstract" TEXT;


ALTER TABLE "MixSub" ADD COLUMN "Title" TEXT;


SELECT *
FROM "MixSub"
LIMIT 10;


SELECT COUNT(*) AS "TODO"
FROM "MixSub"
WHERE "OriginalAbstract" NOT LIKE '%.'
    AND "BetterHighlight" IS NULL;


UPDATE "MixSub"
SET "BetterHighlight" = '',
    "BetterAbstract" = ''
WHERE "Filename" = '';


SELECT COUNT(*)
FROM
    (SELECT "PII",
            "OriginalHighlight",
            "BetterHighlight",
            LENGTH("BetterHighlight")::DECIMAL/LENGTH("OriginalHighlight") AS "RATIO"
     FROM "MixSub"
     WHERE LENGTH("BetterHighlight") >= 1.05 * LENGTH("OriginalHighlight")
     ORDER BY "RATIO" DESC);


SELECT "PII",
       "OriginalHighlight",
       "BetterHighlight",
       LENGTH("BetterHighlight")::DECIMAL/LENGTH("OriginalHighlight") AS "RATIO"
FROM "MixSub"
WHERE "PII" = 'S0142941819313339';

-- UPDATE "MixSub"
-- SET "BetterHighlight" = 'CARPET cosmic ray detector was installed at Riyadh (cut off rigidity Rc=14.4GV) Saudi Arabia.  This is a unique location for monitoring and studying the variations of CRs in the equatorial region.  Measurements from such place will provide the research community with useful information about the cosmic ray properties and variations.  One of the main goal of this detector is to study the CR variations and investigate their correlations with solar activity and atmospheric phenomena.  The detector performance was tested and showed comparable results to our existing 1m2 scintillator and multi-wire detectors.  Short term periodicities of the CR recorded by CARPET were investigated and found to be in good agreement with those reported by different researchers.'
-- WHERE "PII" = 'S1364682620300146';

SELECT "PII",
       "OriginalHighlight",
       "BetterHighlight"
FROM "MixSub"
WHERE (LENGTH("BetterHighlight") >= 1.1 * LENGTH("OriginalHighlight"));


SELECT COUNT(*)
FROM
    (SELECT "PII",
            LENGTH("BetterHighlight")::DECIMAL/ LENGTH("OriginalHighlight") AS "RATIO"
     FROM "MixSub"
     WHERE LENGTH("BetterHighlight") >= 1.1 * LENGTH("OriginalHighlight")
         AND "HallucinatedHighlightEntities" IS NULL
     ORDER BY "RATIO");


SELECT "PII",
       LENGTH("BetterHighlight")::DECIMAL/ LENGTH("OriginalHighlight") AS "RATIO"
FROM "MixSub"
WHERE ("BetterHighlight" IS NULL
       AND "OriginalHighlight" IS NULL)
    OR (LENGTH("BetterHighlight") >= 1.1 * LENGTH("OriginalHighlight")
        AND "HallucinatedHighlightEntities" IS NULL)
ORDER BY "RATIO" DESC
LIMIT $1;


SELECT COUNT(*) AS "TODO"
FROM "MixSub"
WHERE ("BetterHighlight" IS NULL
       AND "OriginalHighlight" IS NULL)
    OR (LENGTH("BetterHighlight") >= 1.1 * LENGTH("OriginalHighlight")
        AND "HallucinatedHighlightEntities" IS NULL);


SELECT "PII",
       "OriginalHighlight",
       "BetterHighlight"
FROM "MixSub"
WHERE "PII" IN ('S0021967320300078',
                'S0045206819321042',
                'S0021967320302867',
                'S0021967320301175');


SELECT "PII",
       "OriginalHighlight",
       "BetterHighlight",
       LENGTH("BetterHighlight")::DECIMAL/LENGTH("OriginalHighlight") AS "RATIO"
FROM "MixSub"
WHERE (LENGTH("BetterHighlight") >= 1.1 * LENGTH("OriginalHighlight")
       AND "HallucinatedHighlightEntities" IS NULL)
ORDER BY "RATIO" DESC;


SELECT "OriginalHighlight"
FROM "MixSub"
WHERE COALESCE(TRIM("OriginalHighlight"), '') IN ('',
                                                  'ADDED_MANUALLY',
                                                  'NOT_AVAILABLE');

-- GROUP BY "OriginalHighlight";

SELECT COUNT(*)
FROM "MixSub"
WHERE "MixSub"."Title" IS NOT NULL;


SELECT COUNT(*)
FROM "MixSub"
WHERE LENGTH("BetterHighlight") >= 1.05 * LENGTH("OriginalHighlight")
    AND LENGTH("BetterHighlight") < 1.1 * LENGTH("OriginalHighlight")
    AND "HallucinatedHighlightEntities" IS NULL;

-- AND "HallucinatedHighlightEntities" IS NOT NULL;

SELECT COUNT(*)
FROM "MixSub"
WHERE ("Abstract" = 'ADDED_MANUALLY'
       OR "Abstract" NOT LIKE '%.')
    AND (COALESCE(TRIM("BetterHighlight"), 'NOT_AVAILABLE') = 'NOT_AVAILABLE'
         OR COALESCE(TRIM("BetterAbstract"), 'NOT_AVAILABLE') = 'NOT_AVAILABLE');


SELECT COUNT(*)
FROM "MixSub"
WHERE "OriginalAbstract" NOT LIKE '%.'
LIMIT 5;


SELECT COUNT(*)
FROM "MixSub"
WHERE "OriginalAbstract" NOT LIKE '%.'
    AND "BetterAbstract" IS NOT NULL;


SELECT COUNT(DISTINCT "PII")
FROM "MixSub"
WHERE "OriginalAbstract" NOT LIKE '%.'
    AND ("BetterHighlight" IS NULL
         OR "BetterAbstract" IS NULL);


SELECT COUNT(*)
FROM "MixSub"
WHERE (COALESCE(TRIM("BetterAbstract"), '') = ''
       AND "OriginalAbstract" NOT LIKE '%.');


SELECT COUNT(*)
FROM "MixSub"
WHERE "ModelGeneratedHighlight" IS NOT NULL;


SELECT *
FROM "MixSubView"
LIMIT 1;


UPDATE "MixSub"
SET "ModelGeneratedHighlight" = NULL
WHERE "PII" = 'S1369527420301065';


SELECT *
FROM "MixSub"
WHERE "PII" = 'S2949719123000146';


SELECT *
FROM "MixSub"
WHERE "PII" = 'S0021967319311707';


select *
from INFORMATION_SCHEMA.COLUMNS
where table_name = 'MixSub';


SELECT *
FROM "MixSub"
WHERE "BetterAbstract" = 'NOT_AVAILABLE'
    OR "BetterHighlight" = 'NOT_AVAILABLE';


SELECT *
FROM
    (SELECT "Filename",
            CASE
                WHEN "BetterAbstract" IS NOT NULL
                     AND "BetterAbstract" != 'NOT_AVAILABLE' THEN "BetterAbstract"
                ELSE "Abstract"
            END AS "Abstract",
            CASE
                WHEN "BetterHighlight" IS NOT NULL
                     AND "BetterHighlight" != 'NOT_AVAILABLE' THEN "BetterHighlight"
                ELSE "Highlight"
            END AS "Highlight"
     FROM "MixSub"
     WHERE "Split" = 'TRAIN')
WHERE "Abstract" NOT LIKE '%.';


SELECT *
FROM "MixSub"
LIMIT 10;


ALTER TABLE "MixSub" ADD COLUMN "HallucinatedHighlight" TEXT;


ALTER TABLE "MixSub"
DROP COLUMN "CorrectHighlight";


ALTER TABLE "MixSub"
DROP COLUMN "ArticleAbstract";

-- PII stands for publisher id identifier
-- https://en.wikipedia.org/wiki/Publisher_Item_Identifier

ALTER TABLE "MixSub" RENAME COLUMN "Filename" TO "PII";


ALTER TABLE "MixSub" RENAME COLUMN "Abstract" TO "OriginalAbstract";


ALTER TABLE "MixSub" RENAME COLUMN "Highlight" TO "OriginalHighlight";


CREATE OR REPLACE VIEW "MixSubView" AS
SELECT "PII",
       CASE
           WHEN COALESCE(TRIM("BetterAbstract", '') != '') THEN "BetterAbstract"
           ELSE "OriginalAbstract"
       END AS "ArticleAbstract",
       CASE
           WHEN COALESCE(TRIM("BetterHighlight", '') != '') THEN "BetterHighlight"
           ELSE "OriginalHighlight"
       END AS "CorrectHighlight",
       "HallucinatedHighlight",
       "Split"
FROM "MixSub";


ALTER TABLE "MixSub" ADD COLUMN "ModelGeneratedHighlight" TEXT;


ALTER TABLE "MixSub" ADD COLUMN "AbstractEntities" JSONB;


ALTER TABLE "MixSub" ADD COLUMN "CorrectHighlightEntities" JSONB;


ALTER TABLE "MixSub" ADD COLUMN "HallucinatedHighlightEntities" JSONB;


SELECT COUNT(*)
FROM "MixSubView";


SELECT *
FROM "MixSubView"
LIMIT 10;


alter table "MixSub"
alter column "OriginalHighlight"
drop not null;


alter table "MixSub"
alter column "OriginalAbstract"
drop not null;


SELECT *
FROM "MixSub"
OFFSET 55
LIMIT 30;


UPDATE "MixSub"
SET "HallucinatedHighlight" = NULL
WHERE "PII" = 'S1369527420301065';


SELECT COUNT(*)
FROM "MixSub"
WHERE "HallucinatedHighlight" = '';


SELECT *
FROM "MixSubView"
WHERE "CorrectHighlight" LIKE '%Markov%';


SELECT "HallucinatedHighlight"
FROM "MixSubView"
WHERE "PII" = 'S1537511020302440';


SELECT t.relname,
       l.locktype,
       page,
       virtualtransaction,
       pid,
       mode,
       granted
FROM pg_locks l,
     pg_stat_all_tables t
WHERE l.relation = t.relid
ORDER BY relation asc;


UPDATE "MixSub"
SET "HallucinatedHighlight" = NULL
WHERE "HallucinatedHighlight" = '';


SELECT COUNT(*)
FROM "MixSub"
WHERE COALESCE(TRIM("HallucinatedHighlight"), '') != '';


SELECT *
FROM "MixSubView"
WHERE "PII" = 'S0968090X19306898';


SELECT COUNT(*)
FROM "MixSub"
WHERE "HallucinatedHighlight" = '';


SELECT COUNT(*)
FROM "MixSub"
WHERE "HallucinatedHighlight" IS NULL;


UPDATE "MixSub"
SET "HallucinatedHighlight" = NULL
WHERE "HallucinatedHighlight" = '';
SELECT w.W AS "WITH_FILTER",
       v.V AS "WITHOUT_FILTER",
       CAST(w.W AS DECIMAL(7, 2)) * 100 / v.V AS "PERCENTAGE",
       100 - PERCENTAGE AS "LOST"
FROM
    (SELECT COUNT(*) AS W
     FROM train
     WHERE LENGTH (Highlights) <= 1024 ) AS w,

    (SELECT COUNT(*) AS V
     FROM train) AS v;

--

SELECT COUNT(*)
FROM "MixSub";


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
WHERE "Abstract" NOT LIKE '%.'
    AND "BetterHighlight" IS NULL;


UPDATE "MixSub"
SET "BetterHighlight" = '',
    "BetterAbstract" = ''
WHERE "Filename" = '';


SELECT *
FROM "MixSub"
WHERE "Filename" = 'S2666498422000771';


SELECT COUNT(*)
FROM "MixSub"
WHERE ("Abstract" = 'ADDED_MANUALLY'
       OR "Abstract" NOT LIKE '%.')
    AND (COALESCE(TRIM("BetterHighlight"), 'NOT_AVAILABLE') = 'NOT_AVAILABLE'
         OR COALESCE(TRIM("BetterAbstract"), 'NOT_AVAILABLE') = 'NOT_AVAILABLE');


SELECT COUNT(*)
FROM "MixSub"
WHERE "Abstract" NOT LIKE '%.';


SELECT COUNT(*)
FROM "MixSub"
WHERE "Abstract" NOT LIKE '%.'
    AND "BetterAbstract" IS NOT NULL;


SELECT COUNT(DISTINCT "Filename")
FROM "MixSub"
WHERE "Abstract" NOT LIKE '%.'
    AND ("BetterHighlight" IS NULL
         OR "BetterAbstract" IS NULL);


SELECT COUNT(*)
FROM "MixSub"
WHERE (COALESCE(TRIM("BetterAbstract"), '') = ''
       AND "Abstract" NOT LIKE '%.');

DESCRIBE "MixSub";


select *
from INFORMATION_SCHEMA.COLUMNS
where table_name = 'MixSub';


SELECT COUNT(*)
FROM "MixSub"
WHERE "Highlight" = 'ADDED_MANUALLY';


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


SELECT COUNT(*)
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
     FROM "MixSub");
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


ALTER TABLE "MixSub" ADD COLUMN "BetterHighlight" TEXT;


ALTER TABLE "MixSub" ADD COLUMN "BetterAbstract" TEXT;


SELECT *
FROM "MixSub"
LIMIT 10;


SELECT COUNT(*) AS "TODO"
FROM "MixSub"
WHERE "Abstract" NOT LIKE '%.'
  AND "BetterHighlight" IS NULL;


UPDATE "MixSub"
SET "BetterHighlight" = "",
    "BetterAbstract" = ""
WHERE "Filename" = "";


SELECT COUNT(DISTINCT "Filename")
FROM "MixSub"
WHERE "Abstract" NOT LIKE '%.'
  AND COALESCE(TRIM("BetterHighlight"), '') <> '';

UPDATE "MixSub" SET "BetterAbstract" = NULL, "BetterHighlight" = NULL
WHERE "Filename" IN (
  'S0308814620309730','S0034425720304776','S0034425720304879','S0034528818315327',
  'S0034528819305090','S0034528819305958','S0034528819306691','S0034528819307003',
  'S0034528819303108','S0034528819303285','S0034528819304667','S003452881930488'
);
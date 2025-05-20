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
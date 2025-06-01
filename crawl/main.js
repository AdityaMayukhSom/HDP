const puppeteer = require("puppeteer");
const dotenv = require("dotenv");
const winston = require("winston");
const { Pool } = require("pg");
const { MongoClient } = require("mongodb");

require("winston-daily-rotate-file");

const transportAll = new winston.transports.DailyRotateFile({
  filename: "application-%DATE%.log",
  dirname: "log/all",
  datePattern: "YYYY-MM-DD-HH-mm",
  frequency: "30m",
  maxSize: "1m",
  maxFiles: "14d",
});

const transportError = new winston.transports.DailyRotateFile({
  level: "error",
  dirname: "log/error",
  filename: "application-error-%DATE%.log",
  datePattern: "YYYY-MM-DD-HH-mm",
  zippedArchive: true,
  frequency: "30m",
  maxSize: "1m",
  maxFiles: "14d",
});

// @ts-ignore
const logger = winston.createLogger({
  transports: [
    //
    // - Write all logs with importance level of `error` or higher to `error.log`
    //   (i.e., error, fatal, but not other levels)
    //
    transportError,
    //
    // - Write all logs with importance level of `info` or higher to `combined.log`
    //   (i.e., fatal, error, warn, and info, but not trace)
    //
    transportAll,
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple()
      ),
    }),
  ],
});

const REQUEST_HEADERS = {
  dnt: "1",
  pragma: "no-cache",
  priority: "u=0, i",
  connection: "keep-alive",
  "sec-ch-ua": '"Chromium";v="136", "Brave";v="136", "Not.A/Brand";v="99"',
  "sec-ch-ua-mobile": "?0",
  "sec-ch-ua-platform": '"Windows"',
  "sec-fetch-dest": "document",
  "sec-fetch-mode": "navigate",
  "sec-fetch-site": "cross-site",
  "sec-fetch-user": "?1",
  "sec-gpc": "1",
  "upgrade-insecure-requests": "1",
  accept:
    "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
  "accept-encoding": "gzip, deflate, br, zstd",
  "accept-language": "en-US,en;q=0.7",
  "cache-control": "no-cache",
};

const USER_AGENT =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36";

const getIndianTime = () => {
  return new Date().toLocaleString("en-IN", {
    timeZone: "Asia/Kolkata",
  });
};

/**
 *
 * @typedef {object} MongoUrlOptions
 * @property {string} [username] - The unique identifier for the file. Required.
 * @property {string} [password] - The title of the content. Optional.
 * @property {string} [host] - A brief summary or abstract of the content. Optional.
 * @property {number} [port] - A specific highlight or key takeaway from the content. Optional.
 * @property {string} [database] - A specific highlight or key takeaway from the content. Optional.
 *
 * @param {MongoUrlOptions} options
 */
const createMongoUrl = (options) => {
  let URL = "mongodb://";
  if (options.username && options.password) {
    URL += options.username + ":" + options.password + "@";
  }
  URL += (options.host || "localhost") + ":";
  URL += (options.port || "27017") + "/";
  URL += options.database || "admin";
  return URL;
};

/**
 * @param {string} seed
 */
const getJournalNameVolume = (seed) => {
  const JOURNAL_PREFIX = "https://www.sciencedirect.com/journal/";
  const parts = seed.substring(JOURNAL_PREFIX.length).split("/");
  return {
    journalName: parts[0],
    journalVolume: parts[2],
  };
};

/**
 * @param {string} pii
 * @param {puppeteer.Page} page
 */
const processPage = (pii, page) => {
  return page.evaluate((pii) => {
    let title = null;
    let titleRoot = document.querySelector("#screen-reader-main-title");
    if (!titleRoot) {
      console.warn(
        `elem with id 'screen-reader-main-title' not found for ${pii}`
      );
      titleRoot = document?.querySelector("h1");
    }
    if (titleRoot instanceof HTMLElement) {
      const titleCont = titleRoot.querySelector(".title-text");
      if (titleCont instanceof HTMLElement) {
        console.info(
          `extracting title from title container (inner tag) in ${pii}`
        );
        title = titleCont.innerText.trim();
      } else {
        console.info(`extracting title from title root (outer tag) in ${pii}`);
        title = titleRoot.innerText.trim();
      }
    } else {
      console.error(`could not found title root in ${pii}`);
    }

    // Fetch the sub-elements from the previously fetched container element
    // Get the displayed text and return it (`.innerText`)
    const container = document.querySelector("#abstracts");
    const hlRoot = container?.querySelector(".author-highlights .list");

    // const hlSel = [
    //   `.list .react-xocs-list-item .list-contents`,
    //   `.list .react-xocs-list-item *:not(.list-label)`,
    //   `.list .react-xocs-list-item`,
    //   `.list`,
    // ];

    let highlight = "";
    if (hlRoot instanceof HTMLElement) {
      highlight = hlRoot.innerText
        .trim()
        .replaceAll("•", ". ")
        .replaceAll("\n", "");

      if (highlight.startsWith(".")) {
        highlight = highlight.slice(1) + ".";
      }

      highlight = highlight.replace(/\.(\s|\.)*\./g, ". ").trim();
    }

    // let highlightNodeList = [];
    // for (const sel of hlSel) {
    //   let tmpHlList = hlRoot?.querySelectorAll(sel);
    //   if (tmpHlList !== undefined && tmpHlList.length > 0) {
    //     // @ts-ignore
    //     highlightNodeList = tmpHlList;
    //     break;
    //   }
    // }

    // let highlight =
    //   highlightNodeList.length > 0
    //     ? [...highlightNodeList]
    //         .filter((n) => n instanceof HTMLElement)
    //         .map((n) => n.innerText)
    //         .map((h) => {
    //           h = h
    //             .trim()
    //             .replaceAll("•", ".")
    //             .replaceAll("\n", "")
    //             .replaceAll(/[^\x00-\x7F]/g, "");
    //           return h.endsWith(".") ? h : h + ".";
    //         })
    //         .join(" ")
    //     : hlRoot instanceof HTMLElement
    //       ? hlRoot.innerText.trim().replaceAll("•", "").replaceAll("\n", "")
    //       : "";

    // if (highlightNodeList.length > 0) {
    //   if (highlight.startsWith("Highlights")) {
    //     highlight = highlight.replace("Highlights", "");
    //   }

    //   highlight = highlight.replaceAll("\n", "");
    // }

    // const abstractNodeList =
    //   container?.querySelector(".abstract")?.querySelectorAll('[id^="spar"]') ||
    //   [];

    // const abstract = [...abstractNodeList]
    //   .filter(nodeHasText)
    //   .map((n) => n.innerText)
    //   .join(" ");

    let abstract = "";
    const absRoot = container?.querySelector(".author");

    if (absRoot instanceof HTMLElement) {
      const headingChildren = absRoot.querySelectorAll(
        ".section-title, h1, h2, h3"
      );

      for (const children of headingChildren) {
        children.parentNode?.removeChild(children);
      }

      abstract = absRoot.innerText
        // .replaceAll(/[^\x00-\x7F]/g, "")
        .replaceAll(/[\uFFFD]|\p{C}/gu, "")
        .trim();
    }

    // const highlightContainer = container?.querySelector(".author-highlights");

    // const highlight = nodeHasText(highlightContainer)
    // ? highlightContainer.innerText
    // : null;

    return {
      title,
      highlight,
      abstract,
    };
  }, pii);
};

/**
 * @param {string[]} piis
 * @param {puppeteer.Browser} browser
 */
const scrapDataFromPIIs = async (piis, browser) => {
  try {
    const page = await browser.newPage();

    // intercept the request with the custom headers
    await page.setExtraHTTPHeaders({ ...REQUEST_HEADERS });
    await page.setJavaScriptEnabled(false);
    await page.setUserAgent(USER_AGENT);

    page.setRequestInterception(true);

    page.on("request", (request) => {
      if (
        request.resourceType() === "script" ||
        request.resourceType() === "font" ||
        request.resourceType() === "stylesheet" ||
        request.resourceType() === "image"
      )
        request.abort();
      else request.continue();
    });

    page.on("console", async (msg) => {
      const msgArgs = msg.args();
      for (let i = 0; i < msgArgs.length; ++i) {
        logger.log(msg.type(), await msgArgs[i].jsonValue());
      }
    });

    /**
     * @typedef {object} DataType
     * @property {string} pii - The unique identifier for the file. Required.
     * @property {string | null} title - The title of the content. Optional.
     * @property {string | null} abstract - A brief summary or abstract of the content. Optional.
     * @property {string | null} highlight - A specific highlight or key takeaway from the content. Optional.
     */

    /**
     * @type {DataType[]}
     */
    const data = [];

    for (const pii of piis) {
      const scienceDirectUri = `https://www.sciencedirect.com/science/article/abs/pii/${pii}`;
      const waybackMachineUri = `https://web.archive.org/web/20250512000000/${scienceDirectUri}`;

      await page.goto(waybackMachineUri, {
        referer: "https://www.google.com/",
        waitUntil: "networkidle2",
      });

      // CAPTCHA handling: If you're expecting a CAPTCHA on the target page, use the following code snippet to check the status of Browser API's automatic CAPTCHA solver
      // const client = await page.createCDPSession();
      // console.log('Waiting captcha to solve...');
      // const { status } = await client.send('Captcha.waitForSolve', {
      //   detectTimeout: 10000,
      // });
      // console.log('Captcha solve status:', status);
      // await page.waitForSelector("#abstracts", {
      //   timeout: 2000,
      // });

      const errorExists = await page
        .$eval("#errorBorder #error, .error-card", (el) => el !== null)
        .catch((e) => {
          // This catch block handles the case where the selector #errorBorder #error is not found at all
          // or any other error during the $eval.
          // If the selector is not found, $eval throws. We want to treat this as "no error element found".
          logger.info(
            `no error element not found on page for file ${pii}, proceeding normally.`
          );
          return false; // No error element found
        });

      const noHighlight = await page
        .$eval("#abstracts .author-highlights .list", (el) => el === null)
        .catch((e) => {
          logger.error(
            `No highlight exists ${pii}, will go to science direct..`
          );
          return true; // No error element found
        });

      const noAbstract = await page
        .$eval("#abstracts .author", (el) => el === null)
        .catch((e) => {
          logger.error(
            `No abstract exists ${pii}, will go to science direct..`
          );
          return true; // No error element found
        });

      let waitTime = 4000;

      if (errorExists || noHighlight || noAbstract) {
        logger.error(
          `Error detected on the initial page for file ${pii}. Navigating to fallback page.`
        );

        await page.goto(scienceDirectUri, {
          referer: "https://www.google.com/",
          waitUntil: "networkidle2",
        });

        // await page.waitForSelector("#abstracts", {
        //   timeout: 2000,
        // });

        waitTime = 6000;
      }

      try {
        const point = await processPage(pii, page);
        // logger.info(JSON.stringify(point, null, 4));
        data.push({ ...point, pii: pii });
      } catch (e) {
        logger.error(`error while processing page for pii ${pii}`, e);
        data.push({
          pii: pii,
          title: null,
          abstract: null,
          highlight: null,
        });
      }

      await new Promise((r) => setTimeout(r, waitTime + Math.random() * 3000));
    }

    return data;
  } catch (error) {
    logger.log(error);
    throw error;
  }
};

/**
 * @param {string} link
 * @param {puppeteer.Browser} browser
 */
const scrapIdentifiersFromSpecificVolumeUrl = async (link, browser) => {
  try {
    const page = await browser.newPage();

    // intercept the request with the custom headers
    await page.setExtraHTTPHeaders({ ...REQUEST_HEADERS });
    await page.setJavaScriptEnabled(false);
    await page.setUserAgent(USER_AGENT);

    page.setRequestInterception(true);

    page.on("request", (request) => {
      if (
        request.resourceType() === "script" ||
        request.resourceType() === "font" ||
        request.resourceType() === "stylesheet" ||
        request.resourceType() === "image"
      )
        request.abort();
      else request.continue();
    });

    page.on("console", async (msg) => {
      const msgArgs = msg.args();
      for (let i = 0; i < msgArgs.length; ++i) {
        logger.log(msg.type(), await msgArgs[i].jsonValue());
      }
    });

    await page.goto(link, {
      referer: "https://www.google.com/",
      waitUntil: "networkidle2",
    });

    return page.evaluate(
      (link, allowedTypes) => {
        const piis = [];
        let prevLink = "";
        let nextLink = "";

        const navElement = document.querySelector("nav.issue-navigation");
        if (navElement instanceof HTMLElement) {
          const prevA = navElement.querySelector(".navigation-pre a");
          if (prevA instanceof HTMLAnchorElement) {
            prevLink = prevA.href;
          }

          const nextA = navElement.querySelector(".navigation-next a");
          if (nextA instanceof HTMLAnchorElement) {
            nextLink = nextA.href;
          }
        }

        const PREFIX_1 = "/science/article/pii/";
        const PREFIX_2 = "https://www.sciencedirect.com/science/article/pii/";

        const articleList = document.querySelectorAll(
          ".article-list > li, .js-article-list > li"
        );

        for (const article of articleList) {
          if (article instanceof HTMLLIElement) {
            const subtypeContainer = article.querySelector(
              ".js-article-subtype"
            );

            if (subtypeContainer instanceof HTMLElement) {
              const subtype = subtypeContainer.innerText.trim().toUpperCase();
              // Correct usage of find(): it needs a callback function
              if (!allowedTypes.includes(subtype)) {
                console.warn("article not in correct subtype");
                continue;
              }

              const articleHeading = article.querySelector("h3");

              if (articleHeading instanceof HTMLHeadingElement) {
                const links = articleHeading.querySelectorAll("a");

                if (links.length === 0) {
                  console.error("no links found in page");
                }

                for (const link of links) {
                  if (link.href.startsWith(PREFIX_1)) {
                    piis.push(link.href.substring(PREFIX_1.length));
                  } else if (link.href.startsWith(PREFIX_2)) {
                    piis.push(link.href.substring(PREFIX_2.length));
                  } else {
                    console.error("could not parse pii from article href");
                  }
                }
              } else {
                console.error("article heading not an html element");
              }
            } else {
              console.error("subtype container not html element");
            }
          } else {
            console.error(`could not parse article title`);
          }
        }

        return {
          prevLink: prevLink,
          currLink: link,
          nextLink: nextLink,
          piis: piis,
        };
      },
      link,
      ["RESEARCH ARTICLE", "REVIEW ARTICLE"]
    );
  } catch (error) {
    console.log(error);
    throw error;
  }
};

/**
 * @param {string} volumeUrl
 * @param {puppeteer.Browser} browser
 */
const crawlJournalVolumes = async (volumeUrl, browser) => {
  const MAX_ERROR_TRY = 3;
  let seed = volumeUrl;
  const { journalVolume } = getJournalNameVolume(volumeUrl);
  const startVolume = journalVolume;
  let lastVolume = journalVolume;
  let errTry = 0;

  /** @type {Map<string, string[]>} */
  const piis = new Map();

  while (seed !== "") {
    logger.info(`starting crawing of ${seed}`);

    try {
      const { journalVolume } = getJournalNameVolume(seed);
      const e = await scrapIdentifiersFromSpecificVolumeUrl(seed, browser);
      logger.info(JSON.stringify(e, null, 4));

      const existingList = piis.get(journalVolume) ?? [];
      piis.set(journalVolume, [...existingList, ...e.piis]);

      lastVolume = journalVolume;

      seed = e.prevLink;
      errTry = 0;
    } catch (e) {
      logger.error(e.stack);
      errTry++;
      if (errTry == MAX_ERROR_TRY) {
        logger.error(`max retries hit for seed ${seed}, breaking`);
        break;
      }
    }

    await new Promise((r) => setTimeout(r, 5000 + Math.random() * 3000));
  }

  return {
    startVolume,
    lastVolume,
    piis: Object.fromEntries(piis),
  };
};

const main = async () => {
  dotenv.config({
    path: ["../.env"],
  });

  try {
    logger.info("Connecting to Browser API...");

    const browser = await puppeteer.launch({
      browser: "chrome",
      executablePath: "/snap/bin/chromium",
      headless: true,
      // browserWSEndpoint: BROWSER_WS,
    });

    // const h = [
    //   "S0034528818315327",
    //   "S0003347220302657",
    //   "S000334722030289X",
    //   "S0009279719320046",
    //   "S0014483519306608",
    //   "S0003347220303018",
    //   "S000927971932004",
    // ];
    // const d = await run(h, browser);
    // console.log(d);
    // return;

    const pool = new Pool({
      user: process.env["PG_USERNAME"],
      password: process.env["PG_PASSWORD"],
      host: process.env["PG_HOST"],
      port: parseInt(process.env["PG_PORT"] || "5432"),
      database: process.env["PG_DATABASE"],
    });

    let updtCnt = 0;
    const batchSize = 4;

    /**
     * @type {import("pg").QueryConfig<[number | string]>}
     */
    const piiQuery = {
      name: "fetch-batched-piis",
      text: `
      SELECT "PII"
      FROM "MixSub"
      WHERE "OriginalAbstract" NOT LIKE '%.'
        AND ("BetterAbstract" IS NULL OR "BetterHighlight" IS NULL)
      LIMIT $1
      `,
      values: [batchSize],
    };

    /**
     * @type {import("pg").QueryConfig<[string, string, string, string]>}
     */
    const updateAbstractHighlightQuery = {
      name: "update-abstract-highlight",
      text: `
      UPDATE "MixSub"
      SET "Title" = $1,
          "BetterAbstract" = $2,
          "BetterHighlight" = $3
      WHERE "PII" = $4;
      `,
    };

    const todoRes = await pool.query(
      `
      SELECT COUNT(*) AS "TODO"
      FROM "MixSub"
      WHERE "OriginalAbstract" NOT LIKE '%.'
        AND ("BetterAbstract" IS NULL OR "BetterHighlight" IS NULL)
      `
    );

    const todoCnt = parseInt(todoRes.rows[0]["TODO"]);
    logger.info(`TOTAL VALUES TO UPDATE :: ${todoCnt}`);

    while (updtCnt < todoCnt) {
      const client = await pool.connect();

      const res = await client.query(piiQuery);

      /**
       * @type {string[]}
       */
      const piis = res.rows.map((r) => r["PII"]);
      if (piis.length == 0) {
        // we are breaking using update count, but this is just extra protection
        logger.info("breaking as piis count is zero");
        break;
      }

      try {
        logger.info(
          `batch started at ${getIndianTime()} with contents ${piis}`
        );

        const data = await scrapDataFromPIIs(piis, browser);

        logger.info(JSON.stringify(data, null, 4));

        for (let i = 0; i < piis.length; ++i) {
          if (!data[i].abstract) {
            logger.error(`found abstract to be falsy for ${piis[i]}`);
            data[i].abstract = ""; // do not set to null, set to empty string
          }

          if (!data[i].highlight) {
            logger.error(`found highlight to be falsy for ${piis[i]}`);
            data[i].highlight = ""; // do not set to null, set to empty string
          }

          try {
            await client.query(
              updateAbstractHighlightQuery, //
              [
                data[i].title, //
                data[i].abstract,
                data[i].highlight,
                piis[i],
              ]
            );
          } catch (e) {
            logger.error("could not update for pii", piis[i]);
          } finally {
            updtCnt += 1;
          }
        }

        logger.info(
          `batch finished at ${getIndianTime()} with contents ${piis}`
        );
      } catch (e) {
        logger.error("error occurred for batch", piis, e);
      }

      client.release();
    }

    await browser.close();
  } catch (e) {
    logger.error(e);
  }
};

const dumpJournalPIIs = async () => {
  try {
    logger.info("Loading dotnet");
    dotenv.config({
      path: ["../.env"],
    });

    logger.info("Connecting to Browser API...");

    const browser = await puppeteer.launch({
      browser: "chrome",
      executablePath: "/snap/bin/chromium",
      headless: true,
      // browserWSEndpoint: BROWSER_WS,
    });

    const pool = new Pool({
      user: process.env["PG_USERNAME"],
      password: process.env["PG_PASSWORD"],
      host: process.env["PG_HOST"],
      port: parseInt(process.env["PG_PORT"] || "5432"),
      database: process.env["PG_DATABASE"],
    });

    logger.info("mongo client creation started");

    const mongo = new MongoClient(
      createMongoUrl({
        username: process.env["MONGO_USERNAME"],
        password: process.env["MONGO_PASSWORD"],
        host: process.env["MONGO_HOST"],
        port: parseInt(process.env["MONGO_PORT"] || "27017"),
        database: process.env["MONGO_DATABASE"],
      }),
      {
        retryWrites: true,
        writeConcern: {
          w: "majority", // Acknowledge from majority of replica set members
          j: true, // Ensure write is committed to the journal
          wtimeout: 5000, // Timeout after 5 seconds if write concern is not met
        },
        // https://notes.dt.in.th/MongoDBAuthSource
        authSource: "admin",
      }
    );

    await mongo.connect();

    logger.info("mongo client created");

    const volBase = [
      "https://www.sciencedirect.com/journal/expert-systems-with-applications-x/vol/1/suppl/C",
      // "https://www.sciencedirect.com/journal/burnout-research/vol/1/issue/2",
    ];

    const DUMP_COLLECTION_NAME = "unchecked-journal-piis";
    const col = mongo.db().collection(DUMP_COLLECTION_NAME);

    for (const vol of volBase) {
      try {
        const { journalName, journalVolume } = getJournalNameVolume(vol);

        const h = await col.findOne({
          journalName: journalName,
        });

        if (h !== null) {
          logger.warn(
            `skipping over journal name ${journalName} as already present in mongo`
          );
          // TODO: add logic to check if the current volume is lesser than the last volume updated volume
          continue;
        }

        logger.info(`starting extraction for journal ${vol}`);
        const d = await crawlJournalVolumes(vol, browser);
        logger.info(
          `extraction finished for journal ${vol}\n${JSON.stringify(d.piis, null, 4)}`
        );

        await col.replaceOne(
          { journalName: journalName },
          {
            journalName: journalName,
            ...d,
          },
          { upsert: true }
        );

        logger.info(`saved extracted data for ${journalName} into mongodb`);
      } catch (e) {
        logger.error(e.stack);
      }
    }

    /**
     * @type {import("pg").QueryConfig<[string, string, string]>}
     */
    const updateAbstractHighlightQuery = {
      name: "insert-abstract-highlight",
      text: `
      INSERT INTO "MixSub" ("PII", "BetterAbstract", "BetterHighlight", "Split")
      VALUES ($1, $2, $3, 'TRAIN') ON CONFLICT DO NOTHING;
      `,
    };

    // let updtCnt = 0;
    // const batchSize = 4;

    // logger.info(`TOTAL VALUES TO UPDATE :: ${todoCnt}`);

    // while (updtCnt < todoCnt) {
    //   const client = await pool.connect();

    //   const res = await client.query(piiQuery);

    //   /**
    //    * @type {string[]}
    //    */
    //   const piis = res.rows.map((r) => r["PII"]);
    //   if (piis.length == 0) {
    //     // we are breaking using update count, but this is just extra protection
    //     logger.info("breaking as piis count is zero");
    //     break;
    //   }

    //   try {
    //     logger.info(
    //       `batch started at ${getIndianTime()} with contents ${piis}`
    //     );

    //     const data = await scrapDataFromPIIs(piis, browser);

    //     logger.info(JSON.stringify(data, null, 4));

    //     for (let i = 0; i < piis.length; ++i) {
    //       if (!data[i].abstract) {
    //         logger.error(`found abstract to be falsy for ${piis[i]}`);
    //         data[i].abstract = ""; // do not set to null, set to empty string
    //       }

    //       if (!data[i].highlight) {
    //         logger.error(`found highlight to be falsy for ${piis[i]}`);
    //         data[i].highlight = ""; // do not set to null, set to empty string
    //       }

    //       try {
    //         await client.query(
    //           updateAbstractHighlightQuery, //
    //           [
    //             data[i].title, //
    //             data[i].abstract,
    //             data[i].highlight,
    //             piis[i],
    //           ]
    //         );
    //       } catch (e) {
    //         logger.error("could not update for pii", piis[i]);
    //       } finally {
    //         updtCnt += 1;
    //       }
    //     }

    //     logger.info(
    //       `batch finished at ${getIndianTime()} with contents ${piis}`
    //     );
    //   } catch (e) {
    //     logger.error("error occurred for batch", piis, e);
    //   }

    //   client.release();
    // }

    await browser.close();
    await mongo.close();
  } catch (e) {
    logger.error(e.stack);
  }
};

dumpJournalPIIs()
  .then(() => process.exit(0))
  .catch((e) => process.exit(1));

const puppeteer = require("puppeteer");
const dotenv = require("dotenv");
const winston = require("winston");
const { Pool } = require("pg");

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

/**
 * @param {string} filename
 * @param {puppeteer.Page} page
 */
const processPage = (filename, page) => {
  return page.evaluate((filename) => {
    let title = null;
    let titleRoot = document.querySelector("#screen-reader-main-title");
    if (!titleRoot) {
      console.warn(
        `elem with id 'screen-reader-main-title' not found for ${filename}`
      );
      titleRoot = document?.querySelector("h1");
    }
    if (titleRoot instanceof HTMLElement) {
      const titleCont = titleRoot.querySelector(".title-text");
      if (titleCont instanceof HTMLElement) {
        console.info(
          `extracting title from title container (inner tag) in ${filename}`
        );
        title = titleCont.innerText.trim();
      } else {
        console.info(
          `extracting title from title root (outer tag) in ${filename}`
        );
        title = titleRoot.innerText.trim();
      }
    } else {
      console.error(`could not found title root in ${filename}`);
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
  }, filename);
};

/**
 * @param {string[]} filenames
 * @param {puppeteer.Browser} browser
 */
const run = async (filenames, browser) => {
  try {
    const page = await browser.newPage();

    const requestHeaders = {
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

    // intercept the request with the custom headers
    await page.setExtraHTTPHeaders({ ...requestHeaders });
    await page.setJavaScriptEnabled(false);
    await page.setUserAgent(
      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36"
    );

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
     * @property {string} filename - The unique identifier for the file. Required.
     * @property {string |null} title - The title of the content. Optional.
     * @property {string | null} abstract - A brief summary or abstract of the content. Optional.
     * @property {string | null} highlight - A specific highlight or key takeaway from the content. Optional.
     */

    /**
     * @type {DataType[]}
     */
    const data = [];

    for (const filename of filenames) {
      const scienceDirectUri = `https://www.sciencedirect.com/science/article/abs/pii/${filename}`;
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
            `no error element not found on page for file ${filename}, proceeding normally.`
          );
          return false; // No error element found
        });

      const noHighlight = await page
        .$eval("#abstracts .author-highlights .list", (el) => el === null)
        .catch((e) => {
          logger.error(
            `No highlight exists ${filename}, will go to science direct..`
          );
          return true; // No error element found
        });

      const noAbstract = await page
        .$eval("#abstracts .author", (el) => el === null)
        .catch((e) => {
          logger.error(
            `No abstract exists ${filename}, will go to science direct..`
          );
          return true; // No error element found
        });

      let waitTime = 4000;

      if (errorExists || noHighlight || noAbstract) {
        logger.error(
          `Error detected on the initial page for file ${filename}. Navigating to fallback page.`
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
        const point = await processPage(filename, page);
        // logger.info(JSON.stringify(point, null, 4));
        data.push({ ...point, filename });
      } catch (e) {
        logger.error(`error while processing page for filename ${filename}`, e);
        data.push({
          filename,
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

const main = async () => {
  dotenv.config({
    path: ["../.env"],
  });

  const getIndianTime = () => {
    return new Date().toLocaleString("en-IN", {
      timeZone: "Asia/Kolkata",
    });
  };

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

    const todoRes = await pool.query(
      `
    SELECT COUNT(*) AS "TODO"
    FROM "MixSub"
    WHERE "Abstract" NOT LIKE '%.'
      AND (COALESCE(TRIM("BetterAbstract"), '') = '' OR COALESCE(TRIM("BetterHighlight"), '') = '')
    `
    );

    const todoCnt = parseInt(todoRes.rows[0]["TODO"]);
    logger.info(`TOTAL VALUES TO UPDATE :: ${todoCnt}`);

    let updtCnt = 0;
    const batchSize = 4;

    /**
     * @type {import("pg").QueryConfig<[number]>}
     */
    const filenameQuery = {
      name: "fetch-batched-file-names",
      text: `
      SELECT "Filename"
      FROM "MixSub"
      WHERE "Abstract" NOT LIKE '%.'
        AND (COALESCE(TRIM("BetterAbstract"), '') = '' OR COALESCE(TRIM("BetterHighlight"), '') = '')
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
      WHERE "Filename" = $4;
      `,
    };

    while (updtCnt < todoCnt) {
      const client = await pool.connect();

      const res = await client.query(filenameQuery);

      /**
       * @type {string[]}
       */
      const filenames = res.rows.map((r) => r["Filename"]);
      if (filenames.length == 0) {
        // we are breaking using update count, but this is just extra protection
        logger.info("breaking as filenames count is zero");
        break;
      }

      try {
        logger.info(
          `batch started at ${getIndianTime()} with contents ${filenames}`
        );

        const data = await run(filenames, browser);

        logger.info(JSON.stringify(data, null, 4));

        for (let i = 0; i < filenames.length; ++i) {
          if (!data[i].abstract) {
            logger.error(`found abstract to be falsy for ${filenames[i]}`);
            data[i].abstract = "NOT_AVAILABLE";
          }

          if (!data[i].highlight) {
            logger.error(`found highlight to be falsy for ${filenames[i]}`);
            data[i].highlight = "NOT_AVAILABLE";
          }

          try {
            await client.query(
              updateAbstractHighlightQuery, //
              [
                data[i].title, //
                data[i].abstract,
                data[i].highlight,
                filenames[i],
              ]
            );
          } catch (e) {
            logger.error("could not update for filename", filenames[i]);
          } finally {
            updtCnt += 1;
          }
        }

        logger.info(
          `batch finished at ${getIndianTime()} with contents ${filenames}`
        );
      } catch (e) {
        logger.error("error occurred for batch", filenames, e);
      }

      client.release();
    }

    await browser.close();
  } catch (e) {
    logger.error(e);
  }
};

main()
  .then(() => process.exit(0))
  .catch((e) => process.exit(1));

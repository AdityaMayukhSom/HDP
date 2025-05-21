const puppeteer = require("puppeteer");
const { Client, Pool, Query } = require("pg");
const dotenv = require("dotenv");
const fs = require("fs");
const winston = require("winston");

// @ts-ignore
const logger = winston.createLogger({
  transports: [
    //
    // - Write all logs with importance level of `error` or higher to `error.log`
    //   (i.e., error, fatal, but not other levels)
    //
    new winston.transports.File({
      filename: "error-outputs.log",
      level: "error",
      dirname: "log",
    }),
    //
    // - Write all logs with importance level of `info` or higher to `combined.log`
    //   (i.e., fatal, error, warn, and info, but not trace)
    //
    new winston.transports.File({
      filename: "combined-outputs.log",
      dirname: "log",
    }),
    new winston.transports.Console({
      format: winston.format.simple(),
    }),
  ],
});

/**
 * @param {string} filename
 * @param {puppeteer.Page} page
 */
const processPage = (filename, page) => {
  return page.$eval("#abstracts", (container) => {
    /**
     * @param {Element | ChildNode | null} node
     */
    const nodeHasText = (node) => {
      return (
        node instanceof HTMLDivElement ||
        node instanceof HTMLParagraphElement ||
        node instanceof HTMLSpanElement ||
        node instanceof HTMLLIElement ||
        node instanceof HTMLDataElement ||
        node instanceof HTMLDListElement ||
        node instanceof HTMLDataListElement
      );
    };

    /**
     * @param {Element | null} parentNode
     */
    function removeHeadingChildren(parentNode) {
      if (!parentNode) {
        logger.warn("Parent node not found.");
        return;
      }

      const headingTags = ["H1", "H2", "H3", "H4", "H5", "H6"]; // Uppercase for nodeName comparison

      // Collect children to remove first, as removing elements during iteration
      // can mess up the NodeList's indices.
      const childrenToRemove = [];
      for (let i = 0; i < parentNode.children.length; i++) {
        const child = parentNode.children[i];
        if (headingTags.includes(child.nodeName)) {
          // nodeName is always uppercase
          childrenToRemove.push(child);
        }
      }

      // Now remove them
      childrenToRemove.forEach((child) => {
        parentNode.removeChild(child);
      });

      // console.log("All heading childs removed from:", parentNode);
    }

    // Fetch the sub-elements from the previously fetched container element
    // Get the displayed text and return it (`.innerText`)
    const highlightNodeList =
      container?.querySelector(".author-highlights")?.querySelectorAll(
        `
          .react-xocs-list-item *:not(.list-label), 
          .list *:not(.list-label)
          `
      ) || [];

    const highlight = [...highlightNodeList]
      .filter(nodeHasText)
      .map((n) => n.innerText)
      .map((h) => {
        h = h
          .trim()
          .replaceAll("•", "")
          .replaceAll("\n", "")
          .replaceAll(/[^\x00-\x7F]/g, "");
        return h.endsWith(".") ? h : h + ".";
      })
      .join(" ");

    // const abstractNodeList =
    //   container?.querySelector(".abstract")?.querySelectorAll('[id^="spar"]') ||
    //   [];

    // const abstract = [...abstractNodeList]
    //   .filter(nodeHasText)
    //   .map((n) => n.innerText)
    //   .join(" ");

    const abstractContainer = container?.querySelector(".author");
    removeHeadingChildren(abstractContainer);

    // const highlightContainer = container?.querySelector(".author-highlights");

    // const highlight = nodeHasText(highlightContainer)
    // ? highlightContainer.innerText
    // : null;

    const abstract = (
      nodeHasText(abstractContainer) ? abstractContainer.innerText : ""
    ).replace(/[^\x00-\x7F]/g, "");

    return {
      filename,
      highlight,
      abstract,
    };
  });
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
      if (request.resourceType() === "script") request.abort();
      else request.continue();
    });

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
        .$eval("#errorBorder #error", (el) => el !== null)
        .catch((e) => {
          // This catch block handles the case where the selector #errorBorder #error is not found at all
          // or any other error during the $eval.
          // If the selector is not found, $eval throws. We want to treat this as "no error element found".
          logger.info(
            `Error element not found on page for file ${filename}, proceeding normally.`
          );
          return false; // No error element found
        });

      let waitTime = 5000;

      if (errorExists) {
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

        waitTime = 15000;
      }

      try {
        const point = await processPage(filename, page);
        // logger.info(JSON.stringify(point, null, 4));
        data.push(point);
      } catch (e) {
        logger.error(`error while processing page for filename ${filename}`, e);
        data.push({ filename: filename, abstract: null, highlight: null });
      }

      await new Promise((r) => setTimeout(r, waitTime + Math.random() * 2000));
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

  try {
    logger.info("Connecting to Browser API...");

    const browser = await puppeteer.launch({
      browser: "chrome",
      executablePath: "/snap/bin/chromium",
      headless: true,
      // browserWSEndpoint: BROWSER_WS,
    });

    const d = await run(
      [
        "S0003347220301986",
        "S0003347220300270",
        "S000334722030244X",
        "S0021961420301294",
      ],
      browser
    );
    console.log(d);
    return;

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
      AND COALESCE(TRIM("BetterAbstract"), '') = ''
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
        AND COALESCE(TRIM("BetterAbstract"), '') = ''
      LIMIT $1
      `,
      values: [batchSize],
    };

    /**
     * @type {import("pg").QueryConfig<[string, string, string]>}
     */
    const updateAbstractHighlightQuery = {
      name: "update-abstract-highlight",
      text: `
      UPDATE "MixSub"
      SET "BetterAbstract" = $1,
          "BetterHighlight" = $2
      WHERE "Filename" = $3;
      `,
    };

    while (updtCnt < todoCnt) {
      const client = await pool.connect();

      const res = await client.query(filenameQuery);

      /**
       * @type {string[]}
       */
      const filenames = res.rows.map((r) => r["Filename"]);

      try {
        logger.info(
          `batch started at ${new Date().toLocaleString(
            "en-IN"
          )} with contents ${filenames}`
        );

        const data = await run(filenames, browser);

        for (let i = 0; i < filenames.length; ++i) {
          if (!data[i].abstract || !data[i].highlight) {
            logger.error(
              `found abstract or highlight to be falsy for filename ${filenames[i]}`,
              data[i].abstract,
              data[i].highlight
            );
            continue;
          }

          try {
            await client.query(
              updateAbstractHighlightQuery, //
              [data[i].abstract, data[i].highlight, filenames[i]]
            );
          } catch (e) {
            logger.error("could not update for filename", filenames[i]);
          }
        }

        logger.info("batch finished at", new Date().toLocaleString("en-IN"));
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
